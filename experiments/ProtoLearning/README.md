# Model overview

## train_icsn_disc_groups_muldec
The iCSN architecture using the Deep InfoMax discriminator instead of the swapping mechanism, with multiple decoders.

## train_icsn_disc_groups_singledec
The iCSN architecture using the Deep InfoMax discriminator instead of the swapping mechanism, with one single decoder.

## train_icsn_groups_muldec
The iCSN architecture training prototype groups individually and using multiple stacked decoders instead of just one.

## train_icsn_groups_singledec
The iCSN architecture training prototype groups individually and using only one decoder.

## train_icsn_unsupervised
The iCSN architecture training unsupervisedly instead of using swapping.

## train_vt_cub
The iCSN swapping architecture with a Vision Transformer as encoder, adapted to train on the CUB dataset.

## train_vt_disc_groups_muldec_class
The iCSN architecture with a Vision Transformer as encoder and the InfoMax discriminator instead of swapping. Groups are trained individually. Reconstruction is done using multiple stacked decoders. Additionally, a classifier is trained.

## train_vt_disc_groups_muldec
The iCSN architecture with a Vision Transformer as encoder and the InfoMax discriminator instead of swapping. Groups are trained individually. Reconstruction is done using multiple stacked decoders.

## train_vt_disc_groups_singledec
The iCSN architecture with a Vision Transformer as encoder and the InfoMax discriminator instead of swapping. Groups are trained individually. Reconstruction is done using one single decoder.

## train_vt_disc
The iCSN architecture with a Vision Transformer as encoder and the InfoMax discriminator instead of swapping. Groups are trained simultaneously. Reconstruction is done using one single decoder.

## train_vt_groups_singledec
The iCSN swapping architecture with a Vision Transformer as encoder. Groups are trained individually. Reconstruction is done using one single decoder.

