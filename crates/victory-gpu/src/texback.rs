//! Textures shaders write that the title's own code then reads on the CPU.
//!
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it
//! stays in the texture's host copy: nothing writes it back. Grand Theft
//! Auto VI's ocean is one such texture — its wave passes write a 256x256
//! height map (`R32_FLOAT`) and a displacement map (`R32G32_FLOAT`), and its
//! CPU code samples both, which is what boats float on; reading memory
//! nothing wrote is why boats sink and the water does not move.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`, off unless set), a texture a list writes is
//! copied into a read-back buffer of ours, and once that list has run its
//! texels are tiled into its title memory in the console's layout, as the
//! console's GPU would have left them.
//!
//! The copy is recorded at the start of the next submission and lands at its
//! end (submissions here wait for their fence), so a title read sees the
//! frame before — as it would before the console's GPU has run.
//!
//! Ported from DexNom/VIctory (`crates/victory-gpu/src/texback.rs`), which
//! measured about 30 places in the title reading these two kinds.

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// `D3D12_RESOURCE_STATE_UNORDERED_ACCESS`.
const UNORDERED_ACCESS: u32 = 0x8;
/// `D3D12_RESOURCE_STATE_COPY_SOURCE`.
const COPY_SOURCE: u32 = 0x800;

/// The kinds copied back: (width, height, format). Empty unless
/// `VICTORY_TEXTURE_BACK` names some.
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let list = std::env::var("VICTORY_TEXTURE_BACK").unwrap_or_default();
        let kinds = parse_kinds(&list);
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds that shaders write are copied back into title memory \
                 (VICTORY_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

/// `<width>x<height>f<format>`, comma separated; what does not parse is
/// left out (a typo turns that kind off rather than the game's start).
fn parse_kinds(list: &str) -> Vec<(u64, u32, u32)> {
    list.split(',')
        .filter_map(|k| {
            let (size, format) = k.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// The textures a submission has written since the last one drained this,
/// by address (a `This` is not `Send`).
static DUE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// A list is about to make `resource` writable by shaders: if it is of a
/// kind copied back, its texels are copied into title memory after it runs.
pub(crate) fn note_write(resource: This, state: u32) {
    // Exactly the state, not a mask of states: a composite that merely
    // includes the bit would copy back textures nothing wrote.
    if state != UNORDERED_ACCESS || kinds().is_empty() || !wanted(resource) {
        return;
    }
    let mut due = DUE.lock().unwrap_or_else(|e| e.into_inner());
    if !due.contains(&(resource as usize)) {
        due.push(resource as usize);
    }
}

/// A texture's copy into a read-back buffer, recorded into a list, and where
/// its texels go once the list has run.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    bytes: usize,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    /// The texture's title memory and its layout's bytes.
    at: usize,
    size: usize,
}

/// Records into `list` the copy of `texture` (title resource `resource`'s
/// host copy) into a read-back buffer.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource) -> Option<Back> {
    let host = host()?;
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc)?.planes[0].layout.clone();
    if layout.desc.swizzle.xor() || !layout.addressable() {
        return None;
    }
    let at = memory as usize;
    let size = layout.size as usize;
    if at == 0 || size == 0 {
        return None;
    }
    // SAFETY: a live texture.
    let d = unsafe { texture.GetDesc() };
    let mips = u32::from(d.MipLevels);
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        mips
    } else {
        mips * u32::from(d.DepthOrArraySize)
    };
    if count == 0 || count > 1024 {
        return None;
    }
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0u64;
    // SAFETY: out arrays of `count` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    let bytes = usize::try_from(total).ok()?;
    if bytes == 0 {
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a read-back buffer.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    let buffer = buffer?;
    crate::list::ensure_state(list, resource, COPY_SOURCE);
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer,
        bytes,
        footprints,
        rows,
        mips,
        layout,
        at,
        size,
    })
}

/// A list of ours with the copies of every texture written since the last
/// submission, and where each one's texels go. None if none is due.
pub(crate) fn prepare() -> Option<(HostList, Vec<Back>)> {
    let due: Vec<usize> = std::mem::take(&mut *DUE.lock().unwrap_or_else(|e| e.into_inner()));
    if due.is_empty() {
        return None;
    }
    let list = host()?.new_list().ok()?;
    let mut back = Vec::new();
    for resource in due {
        let resource = resource as This;
        let Some(texture) = crate::resource::host_resource(resource) else {
            continue;
        };
        match record(&list, resource, texture) {
            Some(b) => back.push(b),
            None => once!(?resource, "a texture of a copied-back kind is not copied back"),
        }
    }
    list.close();
    (!back.is_empty()).then_some((list, back))
}

/// Once the list that recorded `back` has run: its texels, tiled into the
/// title's layout, into the texture's title memory.
pub(crate) fn land(back: &Back) {
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    let mut tiled = vec![0u8; back.size];
    // Bytes of title memory the subresources do not cover stay as they are.
    crate::host::copy_mapped(tiled.as_mut_ptr(), back.at as *const u8, back.size, false);
    // SAFETY: Map succeeded and the buffer was created with exactly this width.
    let mapped = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), back.bytes) };
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = crate::texture::subresource_len(footprint.Offset, pitch, rows, f.Depth, back.bytes as u64);
        let linear = &mapped[footprint.Offset as usize..][..len];
        if let Err(e) = back
            .layout
            .tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize)
        {
            once!(?e, mip, slice, "a texture copied back could not be tiled");
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    let _writing = crate::watch::writing(back.at, back.size);
    crate::host::write_mapped(back.at as *mut u8, tiled.as_ptr(), back.size);
    LANDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Textures copied back since the start.
static LANDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Textures copied back since the last call (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_landed() -> u64 {
    LANDED.swap(0, std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::parse_kinds;

    #[test]
    fn only_well_formed_kinds_are_copied_back() {
        assert_eq!(parse_kinds(""), []);
        assert_eq!(
            parse_kinds("256x256f41,256x256f16, 64x64f16"),
            [(256, 256, 41), (256, 256, 16), (64, 64, 16)]
        );
        // A typo leaves its kind out; the others still work.
        assert_eq!(parse_kinds("256x256f,64x64f16,nonsense"), [(64, 64, 16)]);
        // An unset variable copies nothing back.
        assert!(parse_kinds("0").is_empty());
    }
}
