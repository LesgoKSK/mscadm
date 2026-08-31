# Third-party references and provenance

No third-party source files are redistributed as part of the authoritative
`repro/` implementation. The implementation was cross-checked against these
public references:

- Dumas et al., *A deep generative model for probabilistic energy forecasting
  in power systems: normalizing flows*, and its BSD-2-Clause reference code:
  <https://github.com/jonathandumas/generative-models>
- Hernandez Capel et al., public DDPM energy forecasting implementation:
  <https://github.com/EstebanHernandezCapel/DDPM-Power-systems-forecasting>
- Ordoudis et al., *An Updated Version of the IEEE RTS 24-Bus System for
  Electricity Market and Power System Operation Studies* (2016):
  <https://orbit.dtu.dk/en/publications/an-updated-version-of-the-ieee-rts-24-bus-system-for-electricity->
- MATPOWER IEEE RTS-24 case description and source provenance:
  <https://matpower.org/docs/ref/matpower4.1/case24_ieee_rts.html>

The raw GEFCom2014 data in `Data/` were supplied by the user and are not copied
into generated checkpoints or source packages.
