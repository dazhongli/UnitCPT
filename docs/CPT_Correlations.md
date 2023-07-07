
---
title: "CPT Method"
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
book: true
date: 2023
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

## 

Skin friction within the sand can be determined using Eq.(\ref{eq_fz_sand}). The method implemented is presented in **Section 8.1.4, page 46** of ISO19901-4.

$$ 
\begin{equation} [label@eq_fz_sand]
f(z) = f_L(\sigma_{rc}'+\Delta \sigma_{rd}')\cdot \mathrm{tan}(29^0)
\end{equation}
$$

where 
$\sigma_{rc}'$ refers to the radial confined stress, which can be correlated to the CPT cone tip resistance by Eq.(\ref{eq_rc}).
$$
\begin{equation} [label@eq_rc]
\sigma_{rc}'= \frac{q_c}{44}\cdot A_{re}^{0.3}\cdot \big[\mathrm{max}(1,\frac{h}{D})\big]^{-0.4}
\end{equation}
$$

$$
\begin{equation}
    \Delta\sigma_{rd}' = \frac{q_c}{10}\cdot\bigg(\frac{q_c}{\sigma_v'}\bigg)^{-0.33}\cdot\frac{d_{ref}}{D}
\end{equation}
$$

$$
\begin{equation}
A_{re} = 1-PLR\cdot \bigg(\frac{D_i}{D}\bigg)^2
\end{equation}
$$

where $PLR=1.0$ for typical offshore piles.

\pagebreak
# Reference 

