use clap::Parser;
use cuts_v2::sct_helper::SctHelper;
use faer::Mat;
use image::{open, ImageBuffer, ImageFormat, Rgb};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug, Parser)]
#[command(name = "Angus")]
#[command(about = "Approximates an image with signed cuts", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 'i')]
    input: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    output: std::path::PathBuf,
    /// The width
    #[arg(short = 'c')]
    compression_rate: f64,
    /// The number of blocks to use in delayed matmul
    #[arg(short = 'b')]
    block_size: usize,
    /// The number of tensors to process in parallel
    #[arg(short = 't')]
    threads: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let Args {
        input,
        output,
        compression_rate,
        block_size,
        threads,
    } = Args::try_parse()?;
    let img = open(input)?.into_rgb8();
    let (nrows, ncols) = img.dimensions();
    let (nrows, ncols): (usize, usize) = (nrows as _, ncols as _);
    let mut rgb: [Mat<f32>; 3] = core::array::from_fn(|_| Mat::zeros(nrows, ncols));
    // let mut rmat: Mat<f32> = Mat::zeros(nrows, ncols);
    // let mut gmat: Mat<f32> = Mat::zeros(nrows, ncols);
    // let mut bmat: Mat<f32> = Mat::zeros(nrows, ncols);
    for (row, col, &Rgb([r, g, b])) in img.enumerate_pixels() {
        rgb[0][(row as _, col as _)] = r as _;
        rgb[1][(row as _, col as _)] = g as _;
        rgb[2][(row as _, col as _)] = b as _;
    }
    let nbits_per_color = nrows * ncols * 8;
    let nbits_per_cut = nrows + ncols + 32;
    dbg!(nbits_per_color, nbits_per_cut);
    let width = compression_rate * nbits_per_color as f64 / nbits_per_cut as f64;
    let width = width as usize;
    let mut sct = rgb.map(|mat| {
        SctHelper::new(mat.as_ref(), block_size, width)
    });
    let mats: [Mat<f32>; 3] = core::array::from_fn(|c| {
        let mut mat = &mut sct[c];
        let ref mut rng = StdRng::seed_from_u64(0);
        let normalization = (255 * 255 * nrows * ncols) as f32;
        for w in 0..width {
            let cut = mat.cut(rng);
            dbg!(w, (mat.squared_norm_l2() / normalization).sqrt());
        }
        let r = mat.expand();
        r
    });
    let out = ImageBuffer::from_fn(nrows as _, ncols as _, |i, j| {
        let r = mats[0][(i as _, j as _)];
        let g = mats[1][(i as _, j as _)];
        let b = mats[2][(i as _, j as _)];
        Rgb([to_u8(r), to_u8(g), to_u8(b)])
    });
    out.save(output)?;
    Ok(())
}

fn to_u8(x: f32) -> u8 {
    assert!(x.is_finite());
    let x = x.clamp(0.0, 255.0);
    x.round() as _
}
