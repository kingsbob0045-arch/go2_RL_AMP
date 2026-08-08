
def change_seconds_to_hours_fromat(eta_secs: int) -> str:
    eta_secs = int(eta_secs)
    eta_str  = f"{eta_secs // 3600:02d}:{(eta_secs % 3600) // 60:02d}:{eta_secs % 60:02d}"
    return eta_str