# Results summary

baseline ppl=8.3759 (wikitext-2 test, 20x512 chunks; feedback sweep uses 4 chunks)

## rewire

| config | note | ppl | Δppl% | acc | fact | arith | seq | lang | passkey | freeform[0] |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | stock behaviour | 8.376 | +0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| drop_0 | drop rec layer 0 | 2.75e+04 | +327669.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 'S$$xP |
| drop_1 | drop rec layer 1 | 10.444 | +24.7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon, which i |
| drop_2 | drop rec layer 2 | 9.745 | +16.3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The waves crash against the shore, and the  |
| drop_3 | drop attn layer 3 | 9.711 | +15.9 | 0.976 | 0.917 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon, and the |
| drop_4 | drop rec layer 4 | 8.478 | +1.2 | 0.976 | 0.917 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon and the  |
| drop_5 | drop rec layer 5 | 8.750 | +4.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The waves crash against the shore, and the  |
| drop_6 | drop rec layer 6 | 11.623 | +38.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon, the sta |
| drop_7 | drop attn layer 7 | 8.874 | +5.9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark and mysterious place. It is a place where the sun does not shi |
| drop_8 | drop rec layer 8 | 8.950 | +6.9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| drop_9 | drop rec layer 9 | 8.941 | +6.7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The moonlight that falls on the water is no |
| drop_10 | drop rec layer 10 | 9.011 | +7.6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. It is a place where the sun's light cannot  |
| drop_11 | drop attn layer 11 | 8.832 | +5.4 | 0.976 | 1.0 | 1.0 | 1.0 | 1.0 | 0.833 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| drop_12 | drop rec layer 12 | 8.722 | +4.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. It is a place where the sun has set and the |
| drop_13 | drop rec layer 13 | 8.367 | -0.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. It's a place where the sun has set, and the |
| drop_14 | drop rec layer 14 | 8.977 | +7.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| drop_15 | drop attn layer 15 | 8.630 | +3.0 | 0.952 | 1.0 | 1.0 | 1.0 | 1.0 | 0.667 |  a dark, mysterious place. The moonlight is the only source of illumin |
| drop_16 | drop rec layer 16 | 8.588 | +2.5 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| drop_17 | drop rec layer 17 | 8.680 | +3.6 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| drop_31 | drop attn layer 31 | 26.003 | +210.5 | 0.833 | 1.0 | 1.0 | 1.0 | 0.875 | 0.0 |  a place of mystery and wonder, where bioluminescent creatures light u |
| no_attn | remove all 8 full-attention layers | ERR |  |  |  |  |  |  |  | server exited rc=-6 see no_attn.server.log |
| attn_only | keep only the 8 full-attention layers | ERR |  |  |  |  |  |  |  | server exited rc=-6 see attn_only.server.log |
| no_attn_mid | remove attention layers 11,15,19 | 11.104 | +32.6 | 0.762 | 1.0 | 0.7 | 0.833 | 1.0 | 0.0 |  a dark, mysterious place.⏎The ocean is a vast and mysterious place, a |
| attn_phase_shift | every attention layer moved one slot earlier | 71.831 | +757.6 | 0.262 | 0.25 | 0.5 | 0.333 | 0.125 | 0.0 |   10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10   |

## loops

| config | note | ppl | Δppl% | acc | fact | arith | seq | lang | passkey | freeform[0] |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | stock behaviour | 8.376 | +0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| loop2_12_15 | block 12-15 run 2x | 9.411 | +12.4 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark and mysterious place, but the presence of bioluminescent organ |
| loop2_12_15_damp | block 12-15 run 2x, extra passes at 0.5 | 8.770 | +4.7 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon and star |
| rep2_L14 | single rec layer 14 x2 | 9.054 | +8.1 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon and star |
| rep2_L15 | single attn layer 15 x2 | 9.086 | +8.5 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |

## decay

| config | note | ppl | Δppl% | acc | fact | arith | seq | lang | passkey | freeform[0] |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | stock behaviour | 8.376 | +0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| decay_0.25 | scale DeltaNet log-decay in all recurrent layers | 8.801 | +5.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. It is a place where the sun has set and the |
| decay_0.5 | scale DeltaNet log-decay in all recurrent layers | 8.457 | +1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. It is a place where the sun has set and the |
| decay_2.0 | scale DeltaNet log-decay in all recurrent layers | 8.366 | -0.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| decay_4.0 | scale DeltaNet log-decay in all recurrent layers | 8.593 | +2.6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious, and beautiful sight. The ocean at night is a dark |

## scale

| config | note | ppl | Δppl% | acc | fact | arith | seq | lang | passkey | freeform[0] |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | stock behaviour | 8.376 | +0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| scale_attn_0.0 | full-attention layers scaled | 165.798 | +1879.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  often darkly lit, making it difficult for marine biologists to study  |
| scale_attn_0.5 | full-attention layers scaled | 11.441 | +36.6 | 0.905 | 0.917 | 0.7 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. It's easy to forget that the vast majority  |
| scale_rec_0.5 | recurrent layers scaled | 15.977 | +90.8 | 0.952 | 1.0 | 0.9 | 1.0 | 1.0 | 0.833 |  a mysterious and beautiful sight.⏎The ocean at night is a mysterious  |

## feedback

| config | note | ppl | Δppl% | acc | fact | arith | seq | lang | passkey | freeform[0] |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | stock behaviour | 7.407 | +0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| fb_31_0_0.02 | loop: h31(t-1) -> layer 0, rel 0.02 | 7.619 | +2.9 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon and the  |
| fb_31_0_0.1 | loop: h31(t-1) -> layer 0, rel 0.1 | 7.582 | +2.4 | 0.976 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place.⏎The ocean at night is a dark, mysterious pl |
| fb_8_24_0.3 | skip-forward: h8(t-1) -> layer 24, rel 0.3 | 7.698 | +3.9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |  a dark, mysterious place. The only light comes from the moon and the  |

## state surgery

| name | mode | alpha | layers | hit | output |
|---|---|---|---|---|---|
| dissoc_code_zero_all | zero | 0.5 | 24 layers | True |  7391⏎⏎Q: What is the name |
| dissoc_code_freeze_all | freeze | 0.5 | 24 layers | False | ⏎What is this is this is this is this is this |
| dissoc_code_zero_early | zero | 0.5 | 9 layers | True |  7391⏎⏎Note to self: my name |
| dissoc_code_freeze_early | freeze | 0.5 | 9 layers | False |  What is the secret code?⏎A: What is the |
| dissoc_code_zero_mid | zero | 0.5 | 8 layers | True | ⏎⏎7391 |
| dissoc_code_freeze_mid | freeze | 0.5 | 8 layers | False | ⏎A:⏎A:⏎A:⏎A: |
| dissoc_code_zero_late | zero | 0.5 | 7 layers | True | ⏎⏎7391 |
| dissoc_code_freeze_late | freeze | 0.5 | 7 layers | True | ⏎⏎7391.⏎⏎**Note: This is |
| dissoc_code_overwrite_neutral | transplant | 0.5 | 24 layers | True | ⏎⏎7391 |
| dissoc_name_zero_all | zero | 0.5 | 24 layers | True |  Zorbak⏎Q: What is my favorite color? |
| dissoc_name_freeze_all | freeze | 0.5 | 24 layers | False | ⏎What is this is this is this is this is this |
| dissoc_name_zero_early | zero | 0.5 | 9 layers | True |  Zorbak⏎Q: What is my favourite colour? |
| dissoc_name_freeze_early | freeze | 0.5 | 9 layers | False |  What is my name?⏎A: What is my name |
| dissoc_name_zero_mid | zero | 0.5 | 8 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| dissoc_name_freeze_mid | freeze | 0.5 | 8 layers | False | ⏎A:⏎A:⏎A:⏎A: |
| dissoc_name_zero_late | zero | 0.5 | 7 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| dissoc_name_freeze_late | freeze | 0.5 | 7 layers | True |  Zorbak.⏎Q: What is my favourite colour |
| dissoc_name_overwrite_neutral | transplant | 0.5 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| style_french_blend_0.0 | blend | 0 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_french_blend_0.5 | blend | 0.5 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_french_blend_1.0 | blend | 1 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input text: "Today the w |
| style_french_blend_2.0 | blend | 2 | 24 layers |  | ⏎Aujourd'hui, il fait beau et il y a du soleil.⏎Write one sentence about the streets:⏎Les  |
| style_french_blend_4.0 | blend | 4 | 24 layers |  |  Bonjour ! Je m'appelle Marie. J'aime beaucoup le fromage et le vin.⏎J'adore le fromage de |
| style_french_diff_2.0 | blend | -2 | 24 layers |  | ⏎⏎ weather is mild and the streets are quiet. weather is mild and the streets are quiet. w |
| style_angry_blend_0.0 | blend | 0 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_angry_blend_0.5 | blend | 0.5 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_angry_blend_1.0 | blend | 1 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_angry_blend_2.0 | blend | 2 | 24 layers |  | ⏎⏎The storm raged with such ferocity that it felt like the sky itself was screaming in pro |
| style_angry_blend_4.0 | blend | 4 | 24 layers |  | ⏎1. The one who is the most likely to be the most likely to be the most likely to be the m |
| style_angry_diff_2.0 | blend | -2 | 24 layers |  | ⏎The weather is mild.⏎Write one sentence about the activity:⏎People walk their dogs in the |
| style_joyful_blend_0.0 | blend | 0 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_joyful_blend_0.5 | blend | 0.5 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_joyful_blend_1.0 | blend | 1 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input context: "Today th |
| style_joyful_blend_2.0 | blend | 2 | 24 layers |  | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **Analyze the Request:**⏎    *   Input: A short text expr |
| style_joyful_blend_4.0 | blend | 4 | 24 layers |  | ⏎I am so happy, I am so grateful, I am so loved, I am so loved, I am so loved, I am so lov |
| style_joyful_diff_2.0 | blend | -2 | 24 layers |  | ⏎The weather is mild.⏎Write one sentence about the streets:⏎The streets are quiet.⏎Write o |
| recall_code_control_has_fact | blend | 0 | 24 layers | True | ⏎⏎7391 |
| recall_code_control_no_fact | blend | 0 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_all | transplant | 0.5 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_early | transplant | 0.5 | 9 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_mid | transplant | 0.5 | 8 layers | False | ⏎⏎The secret code is **101010 |
| recall_code_transplant_late | transplant | 0.5 | 7 layers | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_blend_0.5 | blend | 0.5 | 24 layers | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_blend_1.5 | blend | 1.5 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_blend_3.0 | blend | 3 | 24 layers | False |  The secret code is 12345.⏎⏎ |
| recall_code_blend_-1.0 | blend | -1 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L0 | transplant | 0.5 | 0 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L1 | transplant | 0.5 | 1 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L2 | transplant | 0.5 | 2 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L4 | transplant | 0.5 | 4 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L5 | transplant | 0.5 | 5 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L6 | transplant | 0.5 | 6 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L8 | transplant | 0.5 | 8 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L9 | transplant | 0.5 | 9 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L10 | transplant | 0.5 | 10 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L12 | transplant | 0.5 | 12 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L13 | transplant | 0.5 | 13 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L14 | transplant | 0.5 | 14 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L16 | transplant | 0.5 | 16 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L17 | transplant | 0.5 | 17 | False | ⏎⏎**There is no secret code.**⏎⏎The text provided |
| recall_code_transplant_L18 | transplant | 0.5 | 18 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L20 | transplant | 0.5 | 20 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L21 | transplant | 0.5 | 21 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L22 | transplant | 0.5 | 22 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L24 | transplant | 0.5 | 24 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L25 | transplant | 0.5 | 25 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L26 | transplant | 0.5 | 26 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L28 | transplant | 0.5 | 28 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L29 | transplant | 0.5 | 29 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_code_transplant_L30 | transplant | 0.5 | 30 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_control_has_fact | blend | 0 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_control_no_fact | blend | 0 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_all | transplant | 0.5 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_early | transplant | 0.5 | 9 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_mid | transplant | 0.5 | 8 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_late | transplant | 0.5 | 7 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_blend_0.5 | blend | 0.5 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_blend_1.5 | blend | 1.5 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_blend_3.0 | blend | 3 | 24 layers | False |  Your name is Q.⏎⏎Q: What is the capital |
| recall_name_blend_-1.0 | blend | -1 | 24 layers | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L0 | transplant | 0.5 | 0 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L1 | transplant | 0.5 | 1 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L2 | transplant | 0.5 | 2 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L4 | transplant | 0.5 | 4 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L5 | transplant | 0.5 | 5 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L6 | transplant | 0.5 | 6 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L8 | transplant | 0.5 | 8 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L9 | transplant | 0.5 | 9 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L10 | transplant | 0.5 | 10 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L12 | transplant | 0.5 | 12 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L13 | transplant | 0.5 | 13 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L14 | transplant | 0.5 | 14 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L16 | transplant | 0.5 | 16 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L17 | transplant | 0.5 | 17 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L18 | transplant | 0.5 | 18 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L20 | transplant | 0.5 | 20 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L21 | transplant | 0.5 | 21 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L22 | transplant | 0.5 | 22 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L24 | transplant | 0.5 | 24 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L25 | transplant | 0.5 | 25 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L26 | transplant | 0.5 | 26 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L28 | transplant | 0.5 | 28 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L29 | transplant | 0.5 | 29 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
| recall_name_transplant_L30 | transplant | 0.5 | 30 | False | ⏎⏎<think>⏎Thinking Process:⏎⏎1.  **An |
