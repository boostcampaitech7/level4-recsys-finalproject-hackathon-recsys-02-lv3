import { useUserInfoContext } from "~/utils/userInfoContext";
import defaultImage from "~/assets/defaultUser.png";
import { css } from "@emotion/react";

export const ProfileImage = () => {
  const { userInfo } = useUserInfoContext("ProfileImage");
  const profileImage = userInfo.profileImage ?? defaultImage;
  return (
    <div
      css={css({
        marginLeft: 10,
        marginBottom: 10,
        width: 80,
        height: 80,
        overflow: "hidden",
      })}
    >
      <img
        src={profileImage}
        css={css({
          width: "100%",
          height: "100%",
          border: "solid",
          borderWidth: "5px",
          borderRadius: "50%",
          borderColor: "#1ED760",
          objectFit: "cover",
          objectPosition: "center",
        })}
      ></img>
    </div>
  );
};
