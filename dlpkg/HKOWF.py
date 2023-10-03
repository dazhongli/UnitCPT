from pile import PipePile
import soil


gamma_e_UMD = 6.6  # submerg
gamma_e_LMD = 7.5
gamma_e_sand = 9.0
can_17 = PipePile(17, 0.09, 19.4)
can_19 = PipePile(19, 0.09, 19.4)
soils = soil.Stratum(bottom_level=[0, 9.2, 9.2, 10.7, 10.7, 50],
                     drainage=['UD', 'UD', 'UD', 'UD', 'Dr', 'Dr'],
                     gamma=[gamma_e_UMD+10, gamma_e_UMD+10,
                            gamma_e_LMD+10, gamma_e_LMD+10,
                            gamma_e_sand+10, gamma_e_sand+10])