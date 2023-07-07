---
title: "Ground Water Monitoring Data at North Tower"
author: "Dazhong Li"
fontsize: 9pt
titlepage: false
colorlinks: false
geometry: "left=2cm,right=2cm,top=1.8cm,bottom=1.8cm"
toc: true
lof: false
linestretch: 1.0
papersize: a4
bibliography: "bib.bib"

---

# Introduction

BRD, is currently assuming that both anchorage sites is dry jand all design[^1] is based on the assumption that ground water table at anchorage site is low. We were asked by HyD to review the available ground water monitoring at the locations[@hoek2019hoek,p.33].


# Review of the Existing Data 

A summary of the existing ground water monitoring data for North Anchorage is presented in Table 4.4 of Feasibility Study – Preliminary Geotechnical Appraisal Report, which suggests ground water table is located 10~20m below the ground level. Review of the piezometer data available at the South Anchorage suggests a ground level of 30~40m below existing ground level.  However, it should be noted that the available data only covers a monitoring period of 1 week after the installation, and the inferred ground water table may have been affected by the response tests that were conducted within the piezometer pipe. 

![Location Plan of Available Ground Water Monitoring](output/image/FS_location.png){width=600}

![Summary of Monitored Ground Water_1](output/image/FS_summary_1.png){width=600}

![Summary of Monitored Ground Water 2](output/image/FS_summary_2.png){width=600}


# Location of Project Piezometers

**Figure \ref{f_project_location}** shows the location of piezometers installed in the project SI works. Monitoring data span about 2 months from mid March to Mid May. Based on the borehole logs, most of piezometer tips were installed in Grade II or Grade III Rock as shown below in **Table \ref{t_summary_piezometer}**.

Table: Summary of Project Piezometers \label{t_summary_piezometer}

| Piezometer Location | Ground Level | Tip Level | Rock Grade |
| ------------------- | ------------ | --------- | ---------- |
| DH30U               | 63.1         | 41.56     | II         |
| DH30L               | 63.1         | -17.4     | III        |
| DH31                | 61.06        | 41.56     | II         |
| DH32U               | 60.96        | 54.66     | IV         |
| DH32L               | 60.96        | 41.46     | II         |
| DH35                | 61.16        | 51.66     | II         |
| DH36                | 60.92        | 41.42     | III        |
| DH37U               | 111.79       | 62.29     | III        |
| DH37L               | 111.79       | 42.29     | III        |
| DH39                | 61.32        | 51.82     | II         |
| DH40                | 61.04        | 41.54     | II         |
| DH70                | 60.92        | 41.42     | II         |

 ![Typical Arrangement of the Piezometer -R11_DH30 ](output/image/R11_DH30.png){width=600}

 It is noted the response zone of the upper tip is about 18.5m and lower 1.5m. 

![Location Plan of the Piezometers Installed at North Tower \label{f_project_location}](output/image/Location_Plan_of_Piezometer.png){width=600}

# Reading of the Piezometers


`[XXX-XXX]` in the legend refers to `[Ground Level - Tip Level]`

**Figure \ref{f_reading_mbgl}** shows the piezometric levels below the ground level. It is observed that the the inferred apparent ground water table is high, mostly below 1m below ground level. 

![Available Readings of Piezometers (in m below ground)\label{f_reading_mbgl}](output/image/combined_mbgl.png){width=600}


**Figure \ref{f_reading_mpd}** shows the monitored piezometric level, i.e., the ground level - observed piezometric level below. 

![Available Readings of Piezometers (in mPD) \label{f_reading_mpd}](output/image/combined_mpd.png)

From these monitoring data, an initial rise in water within piezometer tubes are noted. This appear to be related to rainfall in `mid Jan` (see **Figure \ref{f_rainfall_jan}**). 

![Rain Fall Records in Jan 2023\label{f_rainfall_jan}](output/image/rainfall_Jan_2023.png){width=600}

![Rain Fall Records in Feb 2023\label{f_rainfall_feb}](output/image/rainfall_Feb_2023.png){width=600}


# Observed Pattern Data Fluctuation 

![Net Drop of Measured Level\label{f_offset}](output/image/offset_data.png){width=600}

**Figure \ref{f_offset}** shows consistent pattern among the piezometers, the dominant period can be found by a Fourier transfer, the power spectrum is shown in **Figure \ref{f_spectrum}**.  The pattern resemble that normally observed sea levels reflecting the tidal response. However, considering the distance between installed station and the shoreline as well as the amplitude, it may not be the tidal effect. A clear pattern with 12 hours and 24 hours are shown the power spectrum(**Figure \ref{f_spectrum}**).

![Power Spectrum of Data\label{f_spectrum}](output/image/DH32L.png){width=600}



[^1]: this is a footnote
