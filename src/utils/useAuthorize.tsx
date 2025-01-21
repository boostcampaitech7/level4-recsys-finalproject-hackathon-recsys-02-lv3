import { useEffect } from "react";
import { typedLocalStorage } from "~/utils/localStorage";
import { useSearchParams } from "react-router-dom";
import { useUserInfoContext } from "./userInfoContext";

const useAuthorize = () => {
  const { userInfo, setUserInfo } = useUserInfoContext("authorized");
  const [searchParams] = useSearchParams();

  const id = userInfo?.id ?? searchParams.get("user_id");
  const profileImage =
    userInfo?.profileImage ?? searchParams.get("user_img_url") ?? undefined;

  useEffect(() => {
    id && typedLocalStorage.set("user_id", Number(id));
    profileImage &&
      profileImage !== "None" &&
      typedLocalStorage.set("user_img_url", String(profileImage));
    console.log("useAuthorize setting", id);
    setUserInfo({
      id: Number(id),
      profileImage:
        profileImage == null || profileImage === "None"
          ? undefined
          : profileImage,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, profileImage]);
};

export default useAuthorize;
