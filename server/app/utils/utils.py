def create_redirect_url(user, embedding, user_img_url=None):
    base_url = "http://localhost:5173"
    if embedding:
        redirect_url = f"{base_url}/home?user_id={user.user_id}"
    else:
        redirect_url = f"{base_url}/onboarding?user_id={user.user_id}"
    if user_img_url:
        redirect_url += f"&user_img_url={user_img_url}"
    return redirect_url