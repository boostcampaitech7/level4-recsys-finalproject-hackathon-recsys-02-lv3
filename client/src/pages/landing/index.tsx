import { useEffect } from "react";
import { css } from "@emotion/react";
import { useNavigate } from "react-router-dom";
import { BASE_URL } from "~/libs/api";
import { typedLocalStorage } from "~/utils/localStorage";
import spotifyLogo from "~/assets/spotifyLogin.png";
import { MobilePadding } from "~/components/MobilePadding";
import { Spacing } from "~/components/Spacing";

export const Component = () => {
  const navigate = useNavigate();
  const dev = import.meta.env.DEV;
  console.log(dev);
  useEffect(() => {
    const userId = Number(typedLocalStorage.get("user_id"));
    if (userId) {
      navigate("/home");
    }
  }, [navigate]);

  return (
    <MobilePadding>
      <Spacing size={500} />
      <button
        onClick={() => (location.href = `${BASE_URL}/login?dev=${dev}`)}
        css={loginCss}
      >
        <img
          src={spotifyLogo}
          css={css({
            height: 35,
            justifyContent: "center",
            paddingRight: 10,
          })}
        />
        <div css={textCss}>Spotify 계정으로 로그인</div>
      </button>
    </MobilePadding>
  );
};

const loginCss = css({
  width: "100%",
  height: "50px",
  background: "#1ED760",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  textAlign: "center",
  borderRadius: 7,
  cursor: "pointer",
  padding: 10,
});

const textCss = css({
  fontFamily: "'Noto Sans KR'",
  fontStyle: "normal",
  fontWeight: "450",
  fontSize: 17,
  justifyContent: "center",
  color: "#000",
  height: "100%",
});
