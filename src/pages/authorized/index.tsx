import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { FullScreenLoader } from "~/components/FullScreenLoader";
import { typedLocalStorage } from "~/utils/localStorage";
import { useUserInfoContext } from "~/utils/userInfoContext";

export const Component = () => {
  const { setUserInfo } = useUserInfoContext("authorized");
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const id = searchParams.get("user_id");
  const profileImage = searchParams.get("user_img_url") ?? undefined;

  useEffect(() => {
    if (id == null) {
      navigate("/");
    } else {
      typedLocalStorage.set("user_id", Number(id));
      typedLocalStorage.set("user_img_url", String(profileImage));
      setUserInfo({ id: Number(id), profileImage });
      navigate("/home");
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <FullScreenLoader />
    </>
  );
};
