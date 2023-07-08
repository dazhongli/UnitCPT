
---
title: "Unified CPT Method for Foundation Design"
author: "Dazhong Li"
fontsize: 11pt
titlepage: true
colorlinks: true
geometry: "left=2cm,right=2cm,top=1.8cm,bottom=1.8cm"
toc: true
lof: true
linestretch: 1.0
papersize: a4
bibliography: "bib.bib"
csl: "cambridge-university-press-author-date.csl"
book: true
date: 2023
link-citations: true
math: "$$"

---
\pagebreak

# CPT Correlation

This document presents the correlation implemented in the current code.
## Basic Engineering Properties

### Unit Weight

@robertson2010soil, refer to say 

$$ 
\begin{equation}
\gamma/\gamma_w = 0.27[\log(R_f)] + 0.36 [\log(q_t/p_a)] + 1.236
\end{equation}
$$



- $\gamma$ = unit weight of soil
- $\gamma_w$ = unit weight of water
- $R_f$ = friction ratio, the ratio of $f_s$ over $q_t$, i.e, $\frac{f_s}{q_t} \times \text{100\%}$
- $q_t$ = corrected cone resistance
- $P_a$ = atmospheric pressure

### Relative Density

> Jamiolkowski (2003), Table 5 on Page 9. 

The original form of relative density can be expressed as Eq.(\ref{eq_fz}).
$$
\begin{equation} [label@eq_fz]
    D_R = \frac{1}{C_2}
\end{equation}
$$

$$
\begin{equation} 
D_r = \frac{1}{3.10}\cdot \ln\bigg[\frac{q_t/P_a}{17.68\cdot(\sigma_{v0}'/P_a))^{0.5}}\bigg]
\end{equation}
$$

### Friction Angle

$$
\begin{equation}
\varphi_p'= 17.6+11\cdot\log_{10}\bigg[\frac{q_t/P_a}{(\sigma_{v0}'/P_a)^{0.5}}\bigg]
\end{equation}
$$

### Small Strain Stiffness $G_0$

$$ 
\begin{equation}
G_0 = 50\cdot \sigma_{atm}\big[(q_t-\sigma_{v0}/\sigma_{atm}\big]^{m*}
\end{equation}
$$

where:
$$
\begin{equation}
    m^*  = \begin{cases}
        0.6 & \text{sand} \\
        0.8 & \text{silt} \\
        1.0 & \text{clay}
    \end{cases}
\end{equation}
$$

## Liquefaction Assessment

## Dissipation Tests

## To-dos
\pagebreak

# Unified CPT Method

Pile capacity can be calculated as:
$$
\begin{equation} 
Q_{r,c} = Q_{f,c} - Q_p = f(z)\cdot A_s + q_{base} \cdot A_{pile}
\end{equation}
$$
## Pile Capacity within Clay

### $\alpha$ Approach
This method will require the input of the undrained shear strength $s_u$. To directly link this 





$$
\begin{equation}
f(z) = \alpha\cdot s_u(z)
\end{equation}
$$

$$
\begin{equation}
\alpha = \begin{cases}
0.5\cdot \psi^{-0.5} & \psi<=1.0 \\
0.5\cdot \psi^{-0.25} & \psi>1.0 \\
\end{cases}
\end{equation}
$$

$$
\begin{equation}
\psi = \frac{s_u}{\sigma_{v0}'(z)}
\end{equation}
$$

## Pile Capacity within Sand

### Skin Friction 

Skin friction within the sand can be determined using Eq.(\ref{eq_fz_sand}). The method implemented is presented in **Section 8.1.4, page 46** of ISO19901-4.

$$ 
\begin{equation} [label@eq_fz_sand]
f(z) = f_L(\sigma_{rc}'+\Delta \sigma_{rd}')\cdot \mathrm{tan}(29^0)
\end{equation}
$$

where 

 - $f_L$ is the loading coefficient, **0.75** for tension actions and 1.0 for compression actions. 
 - 29 is the angle of interface friction used for hte calibration of the method, nothing that factors, such as paint, coating or mill-scale varnish, can negatively affect the interface.
 - $\sigma_{rc}'$ is to the radial confined stress, which can be correlated to the CPT cone tip resistance by Eq.(\ref{eq_rc}).

$$
\begin{equation} [label@eq_rc]
\sigma_{rc}'= \frac{q_c}{44}\cdot A_{re}^{0.3}\cdot \big[\mathrm{max}(1,\frac{h}{D})\big]^{-0.4}
\end{equation}
$$
where

- $\sigma_{rc}'$ is the horizontal effective stress acting on a driven pile at depth $z$, about **two weeks** after driving.
- $\sigma_v'$ is the vertical effective stress ast a depth $z$.
- $A_{re}$ is the effective area ratio, defined in \ref{eq_Are}, is a measure of soil displacement induced by the driven pile and expressed as a fraction of hte soil displacement induced by a close-ended pile (for which $A_{re}=1$). 
$$
\begin{equation}
    \Delta\sigma_{rd}' = \frac{q_c}{10}\cdot\bigg(\frac{q_c}{\sigma_v'}\bigg)^{-0.33}\cdot\frac{d_{ref}}{D}
\end{equation}
$$
where $D$ is the pile outer diameters, $d_{ref}$

$$
\begin{equation} [label@eq_Are]
A_{re} = 1-PLR\cdot \bigg(\frac{D_i}{D}\bigg)^2
\end{equation}
$$
where 

 - *PLR* is the Plug Length Ratio, which has a maximum value of 1.0, defined as the ratio of the plug length ($L_p$) to the pile embedment ($L$), and for which the absence of measurement, *PLR* shall be taken as 1.0 for typical offshore piles.
 - $\Delta \sigma_{rd}'$ is the change in horizontal stress acting at a depth of $z$, arising due to interface shear dilation when the pile is loaded. 
 - $d_ref$ = 0.0356m
 - h is the distance above pile tip at which $f(z)$ acts (=$L-Z$)

### Base Resistance 

When the pile has a length to diameter ratio greater than 5, i.e, $L/D >5.0$, the base resistance $Q_b$ can be calculated in a similar fashion as closed pile

$$ 
\begin{equation}
Q_b = \begin{cases}
q\cdot A_{pile} & L/D >5  (\text{Plugged Case}) \\
q_{c,avg}\cdot A_{re}\cdot A_{pile} & \text{otherwise} (\text{unplugged Case})
\end{cases}
\end{equation}
$$

Note that $A_{pile}$ is the gross area of the tubular piles. $q$ is the maximum bearing pressure at the based of the plugged pile that can be calculated using Eq.(\ref{eq_base}). As the maximum value of $A_{re}$ would be 1.0, the maximum stress allowed would be 0.5 average cone resistance. 

In case of the unplugged case, i.e., $L/D <5.0$, this will be the case for large diameter suction bucket. Assuming $PLR$ is taken as 1.0 for normal offshore piles, the unplugged case will equal to the $q_{avg} * A_{annulus}$.

Base resistance can be calculated as Eq.(\ref{eq_base}). 
$$
\begin{equation}[label@eq_base]
q = [0.12+0.38\cdot A_{re}]\cdot q_{c,avg}
\end{equation}
$$

When 

where 

 - $A_{re}$ is defined in Eq.(\ref{eq_Are}).
 - $q_p$  can normally be adopted as the average $q_c$ below and above the tip level as Eq.(\ref{eq_qp}), however, 

 > lower $q_p$ value shall be adopted where spatial variability in the cone resistance indicate potential design sensitivity[@ISO19901_4,p.44].

$$ 
\begin{equation}[label@eq_qp]
q_{c,avg} = \frac{1}{3D}\int_{-1.5D}^{1.5D}q_c \cdot dz
\end{equation}
$$ 

> @ISO19901_4,p.44. The method does not directly apply to **Gravel**, pile capacity within Gravel may be overpredicted, including the method proposed in the main text

### Skin Friction and End Bearing for Driven Piles in Intermediate Soils

In intermediate soils, CPT would be partially drained. Neither sand or clay method can produce satisfactory results ([@ISO19901_4,p.45])
 - Higher shaft resistance when soils are assumed to be **Clay** than **Sand**
 - Could lead to significantly over-predicted axial capacity compared with offshore load tests
 - In the absence of more definitive criteria, may consider the minimum of the capacity based on Clay or Sand

\pagebreak
# Reference 

