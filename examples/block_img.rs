#![cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]

use block_aligner::scan_block::*;
use block_aligner::scores::*;
use block_aligner::cigar::*;

use image::{Rgb, RgbImage};
use imageproc::drawing::*;
use imageproc::rect::Rect;

use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let mut args = env::args().skip(1);
    let img_path = args.next().expect("Pass in the output file path as the parameter!");

    // uc30_50_60
    //let q = b"MVQATTWKKAIPGLSDEASSSPASELRAPLGGVRAMTMNELTRYSIKEPPSDELGSQLVNLYLQQLHTRYPFLDPAELWRLQKARTPVAHSESGNLSMTQRYGIFKLYMVFAIGATLLQLTNKSAEVSPERFYMTALQHMAAAKVPRTVQNIEAMTLLVVYHLRSASGLGLWYMIGLAMRTCIDLGLHRKNHERGLAPLVIQMHRRLFWTVYSLEIVIAISLGRPLSISERQIDVELPDTISVASVPCPSSPGETPVQPTSSNDNLQLANLLFQLRSIEARIHHSIYRTDKPLSALLPKLDKIYKQLEVWRLASIESLPPDGHVLDYPLLLYHRAVRMLIQPFMTILPVSDPYYVLCLRAAGSVCQMHKRLHQTIGYGHSFIAVQTIFVAGVTLLYGLWTQTHLVWSVTLADDLRACSLVLFVMSERAPWVRKYRDAFEVLVDAAMEKLRSGESSLAEMVAVAQTQAQAQSQSQGPRVGQFASGDETMRGPNPDTGPGSSSYGNGNGEHGGESGDVWRLVTELADWIDQDQETTPKWMPNFEALQSLS";
    //let r = b"MTSETQNSVSPPLAMPGAVAVNPRKRGRTAYVADDASSIAYTRALEERVAFLENKLAQVPTPEATTTPRETASNYSVPSGRDKNALSDVVAHVSLGNFEAPAYVGPSSGLSLALNLGEMVQATVWNKMLPDIQDGTTGNQANCINPSPRCITVEDLLAHSVKEPPSDEQGSQMLKAYTSQLHSKYPFLEPEELWKLHSERLTLAAKPTQTLTRIERFGIFKLYLVYAMGATLVQLTQRGPVLSPEALYITALQHISAARESRTVQNIEAMTLLVMFHLRSTSSHGLWYMIGLAMRTSIDLGLHRAAHEQNLDGPIVQRRRRLFWSVYSLERTIAVSLGRPLSIADNQIDVELPNTSINESPSASVIVGNDITLALVLFKLRRIESKIHHSVYRTDKTLDSLRPKLDRLHQQLKIWRNSLTDWIPTGHPDLNYALLLYNRALRLLIQPFLPILPATDPFYGLCMRAAGDICQAHKRLHQTLDYGHSFIAVQTVFVAGVTLVYGLWTQGNALWSVAVSNDIRACSLVLFVMSERAPWVRKYRDAFEVLVNAAMEKLQDSEAGLAEMASAQMRAGKAPGAADSRGVQNPDLSGNETTTRPMDSSSNQFLMSEDGGIALGEFEGAWPMVAELANWIDQDTEGGSPVWMPNFELLQSLSGTWNE";

    // uc30_80_90
    let q = b"MARGWGNHGPWWLWEDPMLWLGCAAVLTGLVLGGLEQFLWSLPAAVALLLLTPRLAGGQSWAVRCLPCLLVVAFAGYGQWRAEWAMAGSAFTPADEQQAVAVVAQVRSHLVQAYTEAFSSPSGPLLAALVMGQKMAQVPDIIREDFRRAGLSHALAASGFHLSVLLSSVLLMAGQRRLLRLGLGALVILGFIVLAGPQPSVVRAALMAGLGLLLLSLKTRQRPVGVLLVAVIAMLLIAPVWVQSLGFQFSVVATMGLVVSAGPMGEGLSRWLPQRLAMAMAVPLAATCWTIPLQLLHFGRLPLYGIPVNLLLTPILAPLTLTAMVMAPVLLLPPVLTGWLLAVVQPVVVLVVRCFLWMVHGVASLPMAEMPLGQPVPVAAVLLVAACLWWLVPRPTGAAGRPWRRPWLAPVLLLLALALQLQMRFADEIRQLGWSQRRGATTERHRTPDPFLVARHGGRAALISSTARLPFCRRARRELHRLGLDGFDWILVTYRMSKEQRRCWAPLSQQLVRGHDGRLVPGMRLESPGLALVPLSHEAHAYGVTAGSRRARVLIGPAAQRWASSDGGAVLDAWPPLSPVP";
    let r = b"MLWLGCTAVLGGLVLGGLEQFFWSLPVGVGLLLLTPRLGGGRSWGVRCLPCLLVVAFAGYGQWRAELAMAVQAFTPADEQQAVAMVAQVRSHLVQAYSEAFSSPSGPLLAALVMGQKMAQVPDVIREDFRRAGLSHALAASGFHLSVLLSSVLLIAGQRRLLRLGLGALVILGFIALAGPQPSVVRAALMAGLGLLLLSMKTRQRPVGVLLVAVISMLLIAPVWVQSLGFQFSVVATMGLVVSAGPMGEGLSRWLPQRLALAMAIPLAATCWTIPLQLLHFGKLPLYGIPVNLLLTPVLAPLTLTAMMMAPVLLLPPVFTGWLLAVVRPVVVLVVRCFLFMVHGAASLPLAEVPLGQPVPLAAVLLVVACLWWLVPRPGGVAGRPWRRPWLAPVLLLLALAMQVQMRFSDEIRQLGWSRRRSATVERSKEPVPLLVARHQGRAALISATARLSFCRRARKELHRLGLDGFDWILVTYRMSEEQRRCWDALGQQLVRGHDGRLVPGVRLESPGLVLVPLSHEAHAYGVMAGGRRARVLIGPTAQQWAGGDGATVLDAWPPLPSVP";

    let r_padded = PaddedBytes::from_bytes::<AAMatrix>(r, 2048);
    let q_padded = PaddedBytes::from_bytes::<AAMatrix>(q, 2048);
    let run_gaps = Gaps { open: -11, extend: -1 };

    let block_aligner = Block::<_, true, false>::align(&q_padded, &r_padded, &BLOSUM62, run_gaps, 32..=256, 0);
    let blocks = block_aligner.trace().blocks();
    let cigar = block_aligner.trace().cigar(q.len(), r.len());

    let cell_size = 3;
    let img_width = ((r.len() + 1) * cell_size) as u32;
    let img_height = ((q.len() + 1) * cell_size) as u32;
    let bg_color = Rgb([255u8, 255u8, 255u8]);
    let fg_colors = [Rgb([33u8, 155u8, 195u8]), Rgb([198u8, 106u8, 171u8]), Rgb([104u8, 65u8, 127u8])];
    let fg_idx = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() % (fg_colors.len() as u128);
    let fg_color = fg_colors[fg_idx as usize];
    let trace_color = Rgb([255u8, 255u8, 255u8]);
    let mut img = RgbImage::new(img_width, img_height);
    println!("path: {}", img_path);
    println!("img size: {} x {}", img_width, img_height);

    draw_filled_rect_mut(&mut img, Rect::at(0, 0).of_size(img_width, img_height), bg_color);

    for block in &blocks {
        if block.width == 0 || block.height == 0 { continue; }
        let x = (block.col * cell_size) as i32;
        let y = (block.row * cell_size) as i32;
        let width = (block.width * cell_size) as u32;
        let height = (block.height * cell_size) as u32;

        draw_filled_rect_mut(&mut img, Rect::at(x, y).of_size(width, height), fg_color);
        draw_hollow_rect_mut(&mut img, Rect::at(x, y).of_size(width, height), bg_color);
    }

    let mut x = cell_size / 2;
    let mut y = cell_size / 2;
    let vec = cigar.to_vec();

    for op_len in &vec {
        let (next_x, next_y) = match op_len.op {
            Operation::M => (x + op_len.len * cell_size, y + op_len.len * cell_size),
            Operation::I => (x, y + op_len.len * cell_size),
            _ => (x + op_len.len * cell_size, y)
        };
        draw_line_segment_mut(&mut img, (x as f32, y as f32), (next_x as f32, next_y as f32), trace_color);
        x = next_x;
        y = next_y;
    }

    img.save(img_path).unwrap();
}
