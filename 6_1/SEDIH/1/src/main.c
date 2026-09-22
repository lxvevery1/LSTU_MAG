#include "common.h"
#include "decoder.h"
#include "encoder.h"
#include "output.h"
#include "xsrc.h"
#include "xtgt.h"

int main() {
    double pe[POSITIONS][D_MODEL];   // 4x6
    double xsrc[SRC_WORDS][D_MODEL]; // 4x4
    // 1
    posEncoding(pe);
    // 2
    source(pe, xsrc);
    // 3
    target(pe, xtgt);
    // 4: encoder
    double head1[SRC_WORDS][D_K];
    double head2[SRC_WORDS][D_K];
    attentionHead(xsrc, W_Q1, W_K1, W_V1, head1);
    attentionHead(xsrc, W_Q2, W_K2, W_V2, head2);

    PRINT_MATRIX("head1", head1, SRC_WORDS, D_K);
    PRINT_MATRIX("head2", head2, SRC_WORDS, D_K);

    double concat[SRC_WORDS][D_MODEL]; // 4x4
    concatenateHeads(head1, head2, concat);
    PRINT_MATRIX("concat", concat, SRC_WORDS, D_MODEL);

    // Z = Concat * W_O = Concat * I4 = Concat
    double x_attn[SRC_WORDS][D_MODEL];
    attn(xsrc, concat, x_attn);
    PRINT_MATRIX("x_attn", x_attn, SRC_WORDS, D_MODEL);

    double x_out_enc[SRC_WORDS][D_MODEL];
    ffn(x_attn, W_FF1, b1, W_FF2, b2, x_out_enc);
    PRINT_MATRIX("ffn", x_out_enc, SRC_WORDS, D_MODEL);

    // 5: decoder
    double head_dec1[TARGET_WORDS][D_K];
    double head_dec2[TARGET_WORDS][D_K];

    decoderAttentionHead(xtgt, W_Q1, W_K1, W_V1, head_dec1);
    decoderAttentionHead(xtgt, W_Q2, W_K2, W_V2, head_dec2);

    PRINT_MATRIX("decoder head 1", head_dec1, TARGET_WORDS, D_K);
    PRINT_MATRIX("decoder head 2", head_dec2, TARGET_WORDS, D_K);

    double concat_dec[TARGET_WORDS][D_MODEL];

    concatenateDecoderHeads(head_dec1, head_dec2, concat_dec);

    PRINT_MATRIX("decoder concat", concat_dec, TARGET_WORDS, D_MODEL);

    double x_dec_self[TARGET_WORDS][D_MODEL];

    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            x_dec_self[i][j] = xtgt[i][j] + concat_dec[i][j];
        }
    }

    double x_cross_attn[TARGET_WORDS][D_MODEL];

    double head_cross1[TARGET_WORDS][D_K];
    double head_cross2[TARGET_WORDS][D_K];

    crossAttentionHead(x_dec_self, x_out_enc, W_Q_cross1, W_K_cross1,
                       W_V_cross1, head_cross1);

    crossAttentionHead(x_dec_self, x_out_enc, W_Q_cross2, W_K_cross2,
                       W_V_cross2, head_cross2);

    PRINT_MATRIX("cross head 1", head_cross1, TARGET_WORDS, D_K);
    PRINT_MATRIX("cross head 2", head_cross2, TARGET_WORDS, D_K);

    double concat_cross[TARGET_WORDS][D_MODEL];

    concatenateDecoderHeads(head_cross1, head_cross2, concat_cross);
    PRINT_MATRIX("cross concat", concat_cross, TARGET_WORDS, D_MODEL);

    // Cross-attention residual
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            x_cross_attn[i][j] = x_dec_self[i][j] + concat_cross[i][j];
        }
    }
    PRINT_MATRIX("x_cross_attn", x_cross_attn, TARGET_WORDS, D_MODEL);

    double x_out_dec[TARGET_WORDS][D_MODEL];
    decoderFFN(x_cross_attn, W_FF1, b1, W_FF2, b2, x_out_dec);
    PRINT_MATRIX("decoder ffn", x_out_dec, TARGET_WORDS, D_MODEL);

    // 6: logits, softmax, loss for "children"
    double logits[TARGET_WORDS][VOCAB];
    decoderLogits(x_out_dec, W_out, b_out, logits);
    PRINT_MATRIX("logits", logits, TARGET_WORDS, VOCAB);

    double probs[TARGET_WORDS][VOCAB];
    softmaxLogits(logits, probs);
    PRINT_MATRIX("probs", probs, TARGET_WORDS, VOCAB);

    int target_token = 1; // "children" (Дети)
    int target_pos = 0;   // после sos

    double loss = crossEntropyLoss(probs[target_pos], target_token);

    printf("target = %d (\"children\")\n", target_token);
    printf("p(target) = %f\n", probs[target_pos][target_token]);
    printf("loss = %f\n", loss);

    return 0;
}
