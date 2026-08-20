//! Writing a PNG, with no dependencies and no compression.
//!
//! # Why this exists
//!
//! Templates are matched against exact pixels. A screenshot that has been
//! scaled, colour-managed or re-encoded on its way through a screenshot tool is
//! not the image the bot will see at the table, and a template cut from one
//! will miss on the other in ways that look like the reader being unreliable.
//! So captures are written straight from the window buffer, byte for byte.
//!
//! # Why no compression
//!
//! PNG requires its image data to be a zlib stream, but zlib permits blocks
//! that are simply stored verbatim. Using those makes a valid PNG that any
//! reader accepts, in about eighty lines, with no dependency and nothing to go
//! wrong. The cost is size — a full window runs a few megabytes rather than a
//! few hundred kilobytes — which for a handful of reference captures is not a
//! cost worth a compressor.

/// Encodes 24-bit RGB pixels, row-major and top-down, as a PNG.
pub fn encode(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(
        rgb.len(),
        width * height * 3,
        "expected {} bytes for {width}x{height}, got {}",
        width * height * 3,
        rgb.len()
    );

    let mut out = Vec::with_capacity(rgb.len() + rgb.len() / 64 + 1024);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&(width as u32).to_be_bytes());
    header.extend_from_slice(&(height as u32).to_be_bytes());
    // Eight bits per sample, colour type 2 (truecolour), no interlacing.
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &header);

    // Each scanline is prefixed with its filter type. Zero means "none", which
    // is what makes the pixel bytes readable as-is.
    let mut raw = Vec::with_capacity(height * (1 + width * 3));
    for row in 0..height {
        raw.push(0);
        let at = row * width * 3;
        raw.extend_from_slice(&rgb[at..at + width * 3]);
    }

    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Wraps bytes as a zlib stream of stored — uncompressed — deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    // Deflate compression, 32K window, no preset dictionary; the check bits
    // make the two header bytes a multiple of 31.
    let mut out = vec![0x78, 0x01];
    // A stored block carries its length in sixteen bits, so longer input is
    // split across several.
    const BLOCK: usize = 65_535;
    let mut at = 0;
    loop {
        let end = (at + BLOCK).min(data.len());
        let last = end == data.len();
        out.push(u8::from(last));
        let len = (end - at) as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&data[at..end]);
        at = end;
        if last {
            break;
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let mut crc = crc32(kind);
    crc = crc32_from(crc, body);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn adler32(data: &[u8]) -> u32 {
    let (mut low, mut high) = (1u32, 0u32);
    for byte in data {
        low = (low + *byte as u32) % 65_521;
        high = (high + low) % 65_521;
    }
    (high << 16) | low
}

fn crc32(data: &[u8]) -> u32 {
    crc32_from(0, data)
}

/// Continues a CRC over more bytes. `running` is the finished value so far.
fn crc32_from(running: u32, data: &[u8]) -> u32 {
    let mut crc = !running;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            let carry = crc & 1;
            crc >>= 1;
            if carry != 0 {
                crc ^= 0xEDB8_8320;
            }
        }
    }
    !crc
}

/// Reads back a PNG this module wrote.
///
/// # Deliberately not a PNG reader
///
/// It handles exactly what [`encode`] produces: eight-bit truecolour, no
/// interlacing, unfiltered scanlines, and zlib stored blocks. A file from
/// anywhere else will almost certainly be refused, and should be — the purpose
/// is not to open images but to replay recorded sessions through the reader
/// offline.
///
/// That closes the loop the whole debugging effort was missing. Until now a
/// suspected misreading could only be confirmed by asking for another live
/// session, which costs time at a table and reproduces nothing exactly. A
/// recorded frame is the same pixels every time, so a fix can be checked
/// against the exact moment it was meant to fix.
pub fn decode(bytes: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if bytes.len() < 8 || bytes[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return None;
    }
    let (mut width, mut height) = (0usize, 0usize);
    let mut stream: Vec<u8> = Vec::new();
    let mut at = 8;
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize;
        let kind = bytes.get(at + 4..at + 8)?;
        let body = bytes.get(at + 8..at + 8 + len)?;
        match kind {
            b"IHDR" => {
                width = u32::from_be_bytes(body.get(0..4)?.try_into().ok()?) as usize;
                height = u32::from_be_bytes(body.get(4..8)?.try_into().ok()?) as usize;
                // Eight bits per sample, truecolour, no interlacing.
                if body.get(8..13)? != [8, 2, 0, 0, 0] {
                    return None;
                }
            }
            b"IDAT" => stream.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        at += 12 + len;
    }
    if width == 0 || height == 0 || stream.len() < 6 {
        return None;
    }

    // Past the two-byte zlib header, then stored blocks: a flag byte, the
    // length twice (once complemented), then the bytes themselves.
    let mut raw = Vec::with_capacity(height * (1 + width * 3));
    let mut at = 2;
    loop {
        let last = stream.get(at)? & 1;
        let len = u16::from_le_bytes(stream.get(at + 1..at + 3)?.try_into().ok()?) as usize;
        raw.extend_from_slice(stream.get(at + 5..at + 5 + len)?);
        at += 5 + len;
        if last == 1 {
            break;
        }
    }

    let stride = 1 + width * 3;
    if raw.len() < height * stride {
        return None;
    }
    let mut rgb = Vec::with_capacity(width * height * 3);
    for row in 0..height {
        let line = raw.get(row * stride..(row + 1) * stride)?;
        // Only the "none" filter, which is all `encode` writes.
        if line[0] != 0 {
            return None;
        }
        rgb.extend_from_slice(&line[1..]);
    }
    Some((width, height, rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What was written comes back, which is what makes replay meaningful.
    #[test]
    fn a_written_picture_reads_back_identically() {
        for (width, height) in [(3usize, 2usize), (200, 200)] {
            let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
            let png = encode(width, height, &rgb);
            let (w, h, back) = decode(&png).expect("our own file");
            assert_eq!((w, h), (width, height));
            assert_eq!(back, rgb, "{width}x{height} did not survive the round trip");
        }
    }

    #[test]
    fn anything_that_is_not_one_of_ours_is_refused() {
        assert_eq!(decode(b"not a png at all"), None);
        assert_eq!(decode(&[]), None);
        let mut truncated = encode(4, 4, &[9u8; 48]);
        truncated.truncate(20);
        assert_eq!(decode(&truncated), None);
    }

    /// The parts a decoder checks first: signature, dimensions, and that the
    /// chunk lengths walk cleanly to the end.
    #[test]
    fn the_file_is_shaped_like_a_png() {
        let rgb = vec![7u8; 4 * 3 * 3];
        let png = encode(4, 3, &rgb);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

        let mut at = 8;
        let mut kinds = Vec::new();
        while at + 8 <= png.len() {
            let len = u32::from_be_bytes(png[at..at + 4].try_into().expect("four")) as usize;
            let kind = String::from_utf8_lossy(&png[at + 4..at + 8]).to_string();
            // Every chunk carries a CRC the decoder will check.
            let body = &png[at + 8..at + 8 + len];
            let mut crc = crc32(&png[at + 4..at + 8]);
            crc = crc32_from(crc, body);
            let stored = u32::from_be_bytes(
                png[at + 8 + len..at + 12 + len].try_into().expect("four"),
            );
            assert_eq!(crc, stored, "{kind} has a bad checksum");
            kinds.push(kind);
            at += 12 + len;
        }
        assert_eq!(at, png.len(), "the chunks do not reach the end of the file");
        assert_eq!(kinds, vec!["IHDR", "IDAT", "IEND"]);

        let width = u32::from_be_bytes(png[16..20].try_into().expect("four"));
        let height = u32::from_be_bytes(png[20..24].try_into().expect("four"));
        assert_eq!((width, height), (4, 3));
    }

    /// The pixels survive, which is the entire point of not compressing them.
    ///
    /// Found by reading the stored deflate blocks back out: with no filtering
    /// and no compression, the image data is the scanlines verbatim, each
    /// behind a single zero byte.
    #[test]
    fn the_pixels_come_back_unchanged() {
        let width = 3;
        let height = 2;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i * 7) as u8).collect();
        let png = encode(width, height, &rgb);

        // Walk to IDAT, then past the two-byte zlib header and the first
        // stored-block header of five bytes.
        let mut at = 8;
        loop {
            let len = u32::from_be_bytes(png[at..at + 4].try_into().expect("four")) as usize;
            if &png[at + 4..at + 8] == b"IDAT" {
                let body = &png[at + 8..at + 8 + len];
                let raw = &body[2 + 5..body.len() - 4];
                for row in 0..height {
                    let line = &raw[row * (1 + width * 3)..(row + 1) * (1 + width * 3)];
                    assert_eq!(line[0], 0, "row {row} should be unfiltered");
                    assert_eq!(&line[1..], &rgb[row * width * 3..(row + 1) * width * 3]);
                }
                return;
            }
            at += 12 + len;
        }
    }

    /// A capture large enough to need more than one stored block still works.
    #[test]
    fn an_image_past_one_block_is_split_and_stays_valid() {
        let width = 200;
        let height = 200;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
        let png = encode(width, height, &rgb);
        // 200 rows of 601 bytes is 120200, comfortably past the 65535 a single
        // stored block can hold.
        assert!(rgb.len() > 65_535);
        assert!(png.len() > rgb.len());
        assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G']);
    }


    /// Writes a real file so it can be opened by something that is not this
    /// encoder. A PNG that only this module can read is not a PNG.
    #[test]
    #[ignore = "writes a file for a human to look at"]
    fn writes_a_file_a_viewer_can_open() {
        let (width, height) = (160usize, 90usize);
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let at = (y * width + x) * 3;
                rgb[at] = (x * 255 / width) as u8;
                rgb[at + 1] = (y * 255 / height) as u8;
                rgb[at + 2] = if (x / 10 + y / 10) % 2 == 0 { 40 } else { 200 };
            }
        }
        let path = std::env::var("PNG_OUT").unwrap_or_else(|_| "png-check.png".into());
        std::fs::write(&path, encode(width, height, &rgb)).expect("written");
        println!("wrote {path}");
    }

    /// Checked against the value in the zlib specification's own example.
    #[test]
    fn the_checksums_match_known_values() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
