#!/bin/bash

# ------------------ Set hyperparameter tuning permutation options-----------------------------------
OPTIMIZER_G=("0.002")
OPTIMIZER_D=("0.002")
NGF=("16")
NDF=("16")
G_KERNEL=("4")
D_KERNEL=("4")
#CELL_TYPE=("BCC" "SCC" "Benign" "Melanoma")
CELL_TYPE=("MEL")
NZ=("100") 
# "200")
# #OPTIMIZER_ARGS
# LOSS_FUNC=("cross_entropy_loss" "mse_loss" "cross_entropy_weighted_loss")
# LR_SCHEDULER=("StepLR" "LambdaLR" "LinearLR" "ExponentialLR")
# # LR_SCHE_ARGS 



# # # ------------------- Reset the Hyperparameters to default values -----------------------------------
# return_to_default(){
#     D_IND=0
#     # jq --arg a "${PARAMETER_INIT[D_IND]}" '.optimizer.type = $a' ./config/config_hparamtune.json | sponge  ./config/config_hparamtune.json
#     jq --arg a "${OPTIMIZER_G[D_IND]}" '.optimizer_G.args.lr = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json
#     jq --arg a "${OPTIMIZER_D[D_IND]}" '.optimizer_D.args.lr = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json
#     jq --arg a "${NGF[D_IND]}" '.arch_G.args.ngf = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json
#     jq --arg a "${NDF[D_IND]}" '.arch_D.args.ndf = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json

#     jq --arg a "${G_KERNEL[D_IND]}" '.arch_G.args.G_kernel_size = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json
#     jq --arg a "${D_KERNEL[D_IND]}" '.arch_D.args.D_kernel_size = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json
#     # jq --argjson i "${EPOCHS[D_IND]}" '.trainer.epochs = $i' ./config/config_hparamtune.json | sponge  ./config/config_hparamtune.json
#     jq --arg a "" '.run_name = $a' ./config/config_HParamTune_DCGAN.json | sponge  ./config/config_HParamTune_DCGAN.json
# }


# ------------------- Reset the Hyperparameters to default values ----------------------------------- 
# (설명)
# 1. For each hyperparameter group, reset all HParams to default value: 
# 2. For each option for hyperparameter:
# 2. Read the config.json file
# 3. Replace the hyperparameter with the option and replace the file with replace .json
# 4. Run train.py and test.py
# 5. Select the best option for each hyperparameter. 

for TYPE in ${CELL_TYPE[@]}
do
    for OPT_G in ${OPTIMIZER_G[@]}
    do
        for OPT_D in ${OPTIMIZER_D[@]}
        do
            for ngf in ${NGF[@]}
            do
                for ndf in ${NDF[@]}
                do
                    for g_kernel in ${G_KERNEL[@]}
                    do
                        for d_kernel in ${D_KERNEL[@]}
                        do
                            for nz in ${NZ[@]}
                            do
                                NAME="UNCTransGAN_Noise2"_"$TYPE"
                                LOGGINGNAME=_"UNCTransGAN_Noise2"_"$TYPE"+"Glr"_"$OPT_G"+"Dlr"_"$OPT_D"
                                jq --arg a "$NAME" '.name = $a' ./config/config_UNCTransGAN_NoiseM_HParamTune.json | sponge ./config/config_UNCTransGAN_NoiseM_HParamTune.json
                                jq --arg a "$LOGGINGNAME" '.run_name = $a' ./config/config_UNCTransGAN_NoiseM_HParamTune.json | sponge ./config/config_UNCTransGAN_NoiseM_HParamTune.json
                                # jq --arg a "$TYPE" '.data_loader.args.cell_type = $a' ./config/config_WGANGP_HParamTune.json | sponge ./config/config_WGANGP_HParamTune.json
                                # jq '.' ./config/config_HParamTune_DCGAN.json
                                # echo $LOGGINGNAME
                                python3 train.py --device 0,1\
                                -c ./config/config_UNCTransGAN_NoiseM_HParamTune.json\
                                # --opt_G "$OPT_G"\
                                # --opt_D "$OPT_D"\
                                # --ngf "$ngf"\
                                # --ndf "$ndf"\
                                # --g_kernel "$g_kernel"\
                                # --d_kernel "$d_kernel"\
                                # --nz_model "$nz"\
                                # --nz_trainer "$nz"
                            done
                        done
                    done
                done
            done
        done
    done
done
