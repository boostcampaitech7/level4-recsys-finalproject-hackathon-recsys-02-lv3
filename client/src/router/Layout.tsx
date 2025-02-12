import { css } from "@emotion/react";
import { ReactNode, useState } from "react";
import "./reset.css";

import { UserInfo, UserInfoProvider } from "~/utils/userInfoContext";
import { typedLocalStorage } from "~/utils/localStorage";

export const Layout = ({ children }: { children: ReactNode }) => {
  const [userInfo, setUserInfo] = useState<UserInfo>({
    id: typedLocalStorage.get<number>("user_id"),
    profileImage: typedLocalStorage.get<string>("user_img_url"),
  });
  return (
    <UserInfoProvider userInfo={userInfo} setUserInfo={setUserInfo}>
      <div css={wrapperCss}>
        <div css={containerCss}>{children}</div>
      </div>
    </UserInfoProvider>
  );
};

const wrapperCss = css({
  maxWidth: "100%",
  width: "100%",
  padding: 0,
  margin: 0,
  height: "auto",
  backgroundColor: "#262626",
});

const containerCss = css({
  position: "relative",
  maxWidth: 750,
  margin: "0 auto",
  minHeight: "100vh",
  backgroundColor: "#121212",
  color: "#ffffff",
});
