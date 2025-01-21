import { useEffect } from "react";
import { css } from "@emotion/react";
import { useNavigate } from "react-router-dom";
import { BASE_URL } from "~/libs/api";
import { typedLocalStorage } from "~/utils/localStorage";
import spotifyLogo from "~/assets/spotifyLogin.png";

export const Component = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const userId = Number(typedLocalStorage.get("user_id"));
    if (userId) {
      navigate("/home");
    }
  }, [navigate]);
  return (
    <>
      <button onClick={() => (location.href = `${BASE_URL}/login`)}>
        <div css={loginCss}>
          <img
            src={spotifyLogo}
            css={css({
              height: 35,
              justifyContent: "center",
            })}
          />
          <div css={textCss}>Spotify 계정으로 로그인</div>
        </div>
      </button>
    </>
  );
};

const loginCss = css({
  position: "absolute",
  left: "50%",
  top: "70%",
  transform: "translate(-50%)",
  width: "280px",
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
  marginLeft: 5,
  justifyContent: "center",
  color: "#ffffff",
  height: "100%",
});
