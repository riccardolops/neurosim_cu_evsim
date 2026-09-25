// pybind11 module definition for neurosim_cu_esim.

#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated frame-differencing event simulator";
    m.def(
        "evsim",
        &evsim,
        "Generate events from a new grayscale frame (CUDA)",
        py::arg("new_image"),
        py::arg("new_time"),
        py::arg("intensity_state_ub"),
        py::arg("intensity_state_lb"),
        py::arg("event_x_buf"),
        py::arg("event_y_buf"),
        py::arg("event_t_buf"),
        py::arg("event_p_buf"),
        py::arg("contrast_threshold_neg"),
        py::arg("contrast_threshold_pos")
    );
    m.def(
        "evsim_multi",
        &evsim_multi,
        "Generate events from a new grayscale frame, emitting multiple events "
        "per pixel for large log-contrast changes (CUDA)",
        py::arg("new_image"),
        py::arg("new_time"),
        py::arg("prev_time"),
        py::arg("intensity_state_ub"),
        py::arg("intensity_state_lb"),
        py::arg("event_x_buf"),
        py::arg("event_y_buf"),
        py::arg("event_t_buf"),
        py::arg("event_p_buf"),
        py::arg("contrast_threshold_neg"),
        py::arg("contrast_threshold_pos")
    );
    m.def(
        "evsim_voltmeter",
        &evsim_voltmeter,
        "DVS-Voltmeter stochastic event model: Brownian-motion-with-drift "
        "voltage, Inverse-Gaussian/Levy event timestamps (CUDA)",
        py::arg("new_image"),
        py::arg("new_time"),
        py::arg("prev_time"),
        py::arg("base_frame"),
        py::arg("delta_vd_res"),
        py::arg("event_x_buf"),
        py::arg("event_y_buf"),
        py::arg("event_t_buf"),
        py::arg("event_p_buf"),
        py::arg("k1"), py::arg("k2"), py::arg("k3"),
        py::arg("k4"), py::arg("k5"), py::arg("k6"),
        py::arg("seed"),
        py::arg("frame_index")
    );
        m.def(
        "evsim_graca",
        &evsim_graca,
        "Graca & Delbruck (2025) physically-realistic large-signal DVS pixel "
        "model: per-pixel 2nd-order photoreceptor + 1st-order source-follower "
        "advanced in sub-steps, optional shot noise, v2e change detector (CUDA)",
        py::arg("new_image"),
        py::arg("new_time"),
        py::arg("prev_time"),
        py::arg("state"),
        py::arg("event_x_buf"),
        py::arg("event_y_buf"),
        py::arg("event_t_buf"),
        py::arg("event_p_buf"),
        py::arg("Cpd"), py::arg("Cfb"), py::arg("Cpr"), py::arg("Csf"),
        py::arg("Ipr"), py::arg("Isf"),
        py::arg("kappa_fb"), py::arg("kappa_sf"), py::arg("VA"), py::arg("UT"),
        py::arg("ipd_max"), py::arg("ipd_min"), py::arg("intensity_max"),
        py::arg("thr_on"), py::arg("thr_off"), py::arg("refractory_us"),
        py::arg("add_noise"),
        py::arg("stochastic_events"),
        py::arg("seed"),
        py::arg("frame_index"),
        py::arg("init_steady_state")
    );
}
