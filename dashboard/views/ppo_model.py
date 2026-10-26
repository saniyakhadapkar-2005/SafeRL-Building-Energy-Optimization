"""PPO Model page: status, compatibility, inference check, training."""
import numpy as np
import streamlit as st

from dashboard import components, state, ui
from environment.building_env import ACTION_NAMES, MultiFloorBuildingEnv
from rl.model_io import MODEL_PATH, read_meta


def render() -> None:
    components.header()
    ui.flash_messages()
    model, status = state.get_model()
    st.subheader("Model status")
    if status.ready:
        (st.success if not status.is_legacy else st.warning)(f"{status.path.name}: {status.message}")
        meta = status.meta or read_meta(status.path)
        if meta:
            st.markdown(f"- Observation: **{status.observation_version}** · trained steps: **{meta.get('total_timesteps', 'unknown')}** · "
                        f"trained at: {meta.get('trained_at', 'unknown')}\n- Training data: {meta.get('training_data', 'unknown')}")
        else:
            st.caption("No metadata file next to this model (trained outside this version of the project).")
        if ui.button("Run inference check", key="infer_check"):
            env = MultiFloorBuildingEnv(observation_version=status.observation_version)
            obs, _ = env.reset(seed=0)
            action, _ = model.predict(obs, deterministic=True)
            ok = env.action_space.contains(np.asarray(action, dtype=env.action_space.dtype))
            st.write(f"Observation shape {obs.shape}, dtype {obs.dtype} → action {[ACTION_NAMES[int(a)] for a in action]} "
                     f"({'valid' if ok else 'INVALID'} for MultiDiscrete([4,4,4])). The Safety Layer is applied in env.step().")
    else:
        st.error(status.message)
        st.info("Until a compatible model exists you can run the clearly labelled rule-based baseline from Building Simulation.")

    st.subheader("Train / retrain PPO")
    st.caption("Training runs in a separate process and only when you press the button; opening or refreshing the "
               "dashboard never trains. Equivalent command: `python -m rl.train --timesteps 300000`.")
    steps = int(st.number_input("Training timesteps", 20000, 2000000, 300000, 20000,
                                help="≈ 15–20 minutes of CPU per million steps on a typical laptop is a rough guide; 300k is usually a few minutes."))
    tr = state.training_state()
    running = tr.get("state") == "running"
    if ui.button("Train PPO model", key="train_btn", type="primary", disabled=running):
        try:
            state.start_training(steps)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if tr.get("state") == "stale":
        st.warning("A previous training run stopped without finishing (no heartbeat). You can start a new one.")

    def progress() -> None:
        t = state.training_state()
        if t.get("state") == "running":
            total = max(1, int(t.get("total", 1)))
            st.progress(min(1.0, int(t.get("timesteps", 0)) / total), text=f"Training… {t.get('timesteps', 0):,} / {total:,} timesteps")
        elif t.get("state") == "finished":
            st.success(f"Last training finished ({t.get('timesteps', 0):,} timesteps).")
        elif t.get("state") == "failed":
            st.error(f"Last training failed: {t.get('message', 'unknown error')}")

    if running:
        before = tr.get("state")
        def watch() -> None:
            progress()
            if state.training_state().get("state") != "running":
                state.clear_model_cache()
                st.rerun()
        st.fragment(run_every=1.5)(watch)()
    else:
        progress()
