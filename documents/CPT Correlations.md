# CPT Correlation

This document presents the correlation implemented in the current code.
## Basic Engineering Properties

### Unit Weight

> Robertson (2010) 

$$ \gamma/\gamma_w = 0.27[\log(R_f)] + 0.36 [\log(q_t/p_a)] + 1.236$$ 1

- $\gamma$ = unit weight of soil
- $\gamma_w$ = unit weight of water
- $R_f$ = friction ratio, the ratio of $f_s$ over $q_t$, i.e, $\frac{f_s}{q_t} \times \text{100\%}$
- $q_t$ = corrected cone resistance
- $P_a$ = atmospheric pressure

### Relative Density

> Jamiolkowski (2003), Table 5 on Page 9. 
> The original form of relative density can be expressed as $$ D_R = \frac{1}{C_2}$$

$$ D_r = \frac{1}{3.10}\cdot \ln\bigg[\frac{q_t/P_a}{17.68\cdot(\sigma_{v0}'/P_a))^{0.5}}\bigg]$$

### Friction Angle

$$\varphi_p'= 17.6+11\cdot\log_{10}\bigg[\frac{q_t/P_a}{(\sigma_{v0}'/P_a)^{0.5}}\bigg]$$

### Small Strain Stiffness $G_0$

$$ G_0 = 50\cdot \sigma_{atm}\big[(q_t-\sigma_{v0}/\sigma_{atm}\big]^{m*}$$
where:
$$
    m^*  = \begin{cases}
        0.6 & \text{sand} \\
        0.8 & \text{silt} \\
        1.0 & \text{clay}
    \end{cases}
$$

## Liquefaction Assessment

## Dissipation Tests

## To-dos

- [ ] check Gamma
