import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useUserInfoContext } from "./userInfoContext";

const useAuthorize = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { userInfo, setUserInfo } = useUserInfoContext("authorized");

  const saveSession = (id: string, profileImage?: string) => {
    setUserInfo({ id: Number(id), profileImage });
    localStorage.setItem("user_id", id);
    if (profileImage) {
      localStorage.setItem("user_img_url", profileImage);
    }
  };

  useEffect(() => {
    const queryId = searchParams.get("user_id") ?? undefined;
    const queryProfileImage = searchParams.get("user_img_url") ?? undefined;
    const storageId = localStorage.getItem("user_id");
    const storageProfileImage =
      localStorage.getItem("user_img_url") ?? undefined;

    // 1. context에 유저정보 있으면 return
    if (userInfo.id != null) return;

    // 2. 쿼리파라미터에 있으면 유저정보 저장하고 return
    if (queryId != null) {
      saveSession(queryId, queryProfileImage);
      return;
    }

    // 3. 로컬스토리지에 있으면 유저정보 저장하고 return
    if (storageId != null) {
      saveSession(storageId, storageProfileImage);
      return;
    }

    // 4. 셋다 없으면 로그인페이지로
    navigate("/");
  }, []);

  return { id: userInfo.id };
};

export default useAuthorize;
