---
bibliography: [bib.bib]
---

# P-y curves for monotonic actions

## Clay

$$
\begin{equation}
p_u = N_p \cdot s_u  \cdot D
\end{equation}
$$

where:

- $N_p = N_{p0}+ \frac{\gamma' z}{s_u}\le N_{pd}$
- $N_p = 2 N_{p0} \le N_{pd}$

$$
\begin{equation}
N_p = \begin{cases}
   N_{p0} + \frac{\gamma' z}{s_u} \le N_{pd} & \text{if gapping is assumed} \\
   2 \ N_{p0} \le N_{pd} & \text{if no gapping is assumed}
\end{cases}
\end{equation}
$$

$$
\begin{equation}
N_{p0} = N_1 -(1-\alpha_{ave}) - (N_1-N_2) \bigg[1-\bigg(\frac{z}{d\cdot D}\bigg)^{0.6}\bigg]^{1.35}
\end{equation}
$$

where:

- $N_1=12$
- $N_2 = 3.22$
- $d = 16.8 - 2.3\log_10{(\lambda)}\ge 14.5$
- $\lambda = s_{u0} / (s_{u1} D)$
- $\alpha_{ave}$ = the average $\alpha$ over a depth of 20m pile diameter
- $D$ = pile diameter
- $z$ = depth below _original_ seafloor

> Note:
> The is refer to the **DSS strength **
> For normally consolidated condition and $s_u > 15kPa$ for top 10m, **no gap**

For the wedged failure, i.e., $N_p < N_{pd}$, _triaxial extension_ should be used instead of _DSS_ strength.

$$
\begin{equation}
N_{pcor} = c_w N_{p0} + \frac{\gamma' z }{s_u} \le N_{pd}
\end{equation}
$$

where

$$
\begin{equation}
c_w = 1+ \bigg(\frac{s_{uTE}}{s_{uDSS}} -1 \bigg) \ \frac{N_{pd}- N_p}{N_{pd}-N_{p|z=0}}
\end{equation}
$$

where:

- $\frac{s_{uTE}}{S_{uDSS}}=0.9$ based on @ISO19901_4,pp.52

Table:Normalised p-y curves for monotonic actions in clay $I_P$>30% \label{t_py_clay}

| p/pu  |   OCR<=2 |    OCR=4 |   OCR=10 |
| :---: | -------: | -------: | -------: |
|   0   |        0 |        0 |        0 |
| 0.05  |   0.0003 |    0.004 |   0.0005 |
|  0.2  |    0.003 |    0.004 |    0.005 |
|  0.3  |   0.0053 |    0.008 |    0.001 |
|  0.4  |    0.009 |    0.015 |    0.021 |
|  0.5  |    0.014 |    0.024 |    0.034 |
|  0.6  |    0.022 |    0.036 |    0.052 |
|  0.7  |    0.032 |    0.055 |    0.078 |
|  0.8  |     0.05 |    0.084 |     0.12 |
|  0.9  |    0.082 |     0.14 |     0.19 |
| 0.975 |     0.15 |     0.23 |      0.3 |
|  1.0  |     0.25 |      0.3 |      0.4 |
|  1.0  | $\infty$ | $\infty$ | $\infty$ |

## P-y curve for cyclic actions

$$
\begin{equation}
\begin{matrix}
p_{cy} = p_{mod} \cdot p_{mo} \\
y_{cy} = y_{mod} \cdot y_{mo}
\end{matrix}
\end{equation}
$$

$$
\begin{equation}
h_f = \begin{cases}
\frac{p_{mo}}{p_u} - \bigg(\frac{z}{z_{rot}}\bigg)^2 & \text{if $z \le z_{rot}$} \\
\frac{p_{mo}}{p_u} -1 & \text{$z>z_{rot}$}
\end{cases}
\end{equation}
$$

$$
\begin{equation}
N_{eq}  = \bigg( \frac{2}{1-h_f}\bigg)^g \le 25
\end{equation}
$$

$$
\begin{equation}
g = \begin{cases}
1.0 & \text{Gulf of Mexico Condition} \\
1.25 & \text{North Sea Soft Clay Condition}\\
2.5 & \text{North Sea stiff Clay condition}
\end{cases}
\end{equation}
$$

$$
\begin{equation}
p_{mod} = \begin{cases}
1.47 - 0.14\cdot \ln{N_{eq}} & \text{Gulf of Mexico} \\
1.63 - 0.15\cdot \ln{N_{eq}} & \text{North Sea soft Clay} \\
1.45 - 0.17\cdot \ln{N_{eq}} & \text{North Sea stiff Clay}
\end{cases}
\end{equation}
$$


$$
\begin{equation}
y_{mod} = \begin{cases}
1.2 - 0.14\cdot \ln{N_{eq}} & \text{Gulf of Mexico} \\
1.2 - 0.17\cdot \ln{N_{eq}} & \text{North Sea soft Clay} \\
1.2 - 0.17\cdot \ln{N_{eq}} & \text{North Sea stiff Clay}
\end{cases}
\end{equation}
$$

## p-y curves for fatigue action

$$
\begin{equation}
p_{fa} = p_u A_s \cdot \bigg(\frac{y_{fa}}{D}\bigg)^{-B_s}
\end{equation}
$$

where 

- $y_{fa}$ = lateral displacement for fatigue actions
- $A_s$ = 0.45 if $s_u$ < 40kPa, 0.19 otherwise
- $B_s$ = 0.05

## p-y curves for earthquake


## Lateral Capacity for sand

The ultimate lateral resistance of for pile within sand is the minimum of the Eq.(\ref{eq_pr})

$$
\begin{equation}[label@eq_pr]
p_r = \min{(p_{rs},p_{rd})}
\end{equation}
$$

$$
\begin{equation}
p_{rs} = {C_1 z + C_2 D} \cdot \gamma'z
\end{equation}
$$

$$
\begin{equation}
p_{rd} = C_3 \cdot D \cdot  \gamma'
\end{equation}
$$

$$
\begin{equation}
C_1 = \frac{(tan\beta)^2 \tan\alpha }{\tan{(\beta-\phi')}}+ K_0 \bigg(
    \frac{\tan{\phi'}\tan{\beta}}{\cos{\alpha}\tan{(\beta-\phi')}} + \tan{\beta}(\tan{\phi'}\sin{\beta}-\tan{\alpha})\bigg)
\end{equation}
$$

$$
\begin{equation}
C_2 = \frac{\tan{\beta}}{\tan{(\beta-\phi')}} - K_a 
\end{equation}
$$

$$
\begin{equation}
C_3 = K_a ((\tan{\beta})^8-1) + K_0 + \tan{\phi'}(\tan{\beta})^4
\end{equation}
$$
