from pymcd.mcd import Calculate_MCD

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_toolbox = Calculate_MCD(MCD_mode="plain")

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
mcd_value = mcd_toolbox.calculate_mcd('data/clean.wav', 'data/denoised.wav')

print(f'mcd value:{mcd_value}')
