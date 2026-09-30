#pragma once
// Physical two-node photoreceptor model. All quantities are SI.
// Shared by the CUDA kernel and independent host-side validation harness.
#ifdef __CUDACC__
#define GRACA_HD __host__ __device__ __forceinline__
#else
#define GRACA_HD inline
#endif

namespace graca {
constexpr double electron_charge = 1.602176634e-19;
struct Circuit {
    double cpd, cfb, cpr, csf, ipr, isf, kfb, ksf, va, ut;
};
struct OperatingPoint {
    double gs, gm, ga, rout, loop_gain, zmdc;
};
GRACA_HD OperatingPoint operating_point(const Circuit& c, double ipd) {
    OperatingPoint p;
    p.gs = ipd / c.ut;
    p.gm = c.kfb * p.gs;
    p.ga = c.kfb * c.ipr / c.ut;
    p.rout = c.va / (2.0 * c.ipr);
    const double loop = p.ga * p.rout * c.kfb;
    p.loop_gain = loop / (1.0 + loop);
    p.zmdc = p.loop_gain / p.gm;
    return p;
}

// C d[vpd,vpr]/dt + G [vpd,vpr] = [-i_pd, i_pr].
// C = [[Cpd+Cfb,-Cfb],[-Cfb,Cpr+Cfb]],
// G = [[gs_fb,-gm_fb],[gm_amp,1/Rout]].
// The determinant gives the exact second-order denominator (thesis Table 2.1).
// Freeze G at the current LPV operating point for this trapezoidal step.
// force_pd/force_pr are integrated charges, e.g. sqrt(2*q*I*dt)*N(0,1).
GRACA_HD void node_step(const Circuit& c, const OperatingPoint& p, double dt,
                        double force_pd, double force_pr,
                        double& vpd, double& vpr) {
    const double h = 0.5 * dt;
    const double a = c.cpd + c.cfb + h * p.gs;
    const double b = -c.cfb - h * p.gm;
    const double d = -c.cfb + h * p.ga;
    const double e = c.cpr + c.cfb + h / p.rout;
    const double r0 = (c.cpd + c.cfb - h * p.gs) * vpd
                    + (-c.cfb + h * p.gm) * vpr + force_pd;
    const double r1 = (-c.cfb - h * p.ga) * vpd
                    + (c.cpr + c.cfb - h / p.rout) * vpr + force_pr;
    const double determinant = a * e - b * d;
    vpd = (r0 * e - b * r1) / determinant;
    vpr = (a * r1 - d * r0) / determinant;
}
GRACA_HD void signal_step(const Circuit& c, const OperatingPoint& p, double dt,
                          double target_old, double target_new,
                          double& vpd, double& vpr) {
    node_step(c, p, dt, -0.5 * dt * (target_old + target_new) / p.zmdc,
              0.0, vpd, vpr);
}
GRACA_HD double sf_step(const Circuit& c, double dt, double in_old,
                       double in_new, double out_old, double noise_charge = 0.0) {
    const double gs = c.isf / c.ut;
    const double hgs = 0.5 * dt * gs;
    return ((c.csf - hgs) * out_old + c.ksf * hgs * (in_old + in_new)
            + noise_charge) / (c.csf + hgs);
}
// Sampled ideal reset switch: hold the detector reference for any step that
// starts in refractory, including its release endpoint. Release is therefore
// delayed by at most one substep. Endpoint events are similarly quantized.
GRACA_HD int detect_step(double vsf, double step_us, double threshold_on,
                         double threshold_off, double refractory_us,
                         double& reference, double& since_event_us) {
    if (since_event_us < refractory_us) {
        since_event_us += step_us;
        reference = vsf;
        return -1;
    }
    since_event_us += step_us;
    const double difference = vsf - reference;
    const int polarity = difference >= threshold_on ? 1 :
                         (difference <= -threshold_off ? 0 : -1);
    if (polarity >= 0) {
        reference = vsf;
        since_event_us = 0.0;
    }
    return polarity;
}
} // namespace graca
#undef GRACA_HD
