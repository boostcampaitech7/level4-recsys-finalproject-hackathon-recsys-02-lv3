import { useEffect } from "react";
import { css } from "@emotion/react";
import { useNavigate } from "react-router-dom";
import { BASE_URL } from "~/libs/api";
import { typedLocalStorage } from "~/utils/localStorage";
import spotifyLogo from "~/assets/spotifyLogin.png";
import { MobilePadding } from "~/components/MobilePadding";
import { Spacing } from "~/components/Spacing";
import logoImage from "~/assets/logo.png";
import { Title } from "~/components/Title";

export const Component = () => {
  const navigate = useNavigate();
  const dev = import.meta.env.DEV;

  useEffect(() => {
    const userId = Number(typedLocalStorage.get("user_id"));
    if (userId) {
      navigate("/home");
    }
  }, [navigate]);

  return (
    <MobilePadding>
      <Spacing size={200} />
      <div
        css={css({
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        })}
      >
        <img src={logoImage} height={180} width={180} />
      </div>
      <div
        css={css({
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          fontFamily: "'Julius Sans One'",
        })}
      >
        <Title fontSize={30} color={"#72F3A0"}>
          TuneYourShop
        </Title>
      </div>
      <Spacing size={150} />
      <button
        onClick={() => (location.href = `${BASE_URL}/login?dev=${dev}`)}
        css={loginCss}
      >
        <img
          src={spotifyLogo}
          css={css({
            height: 35,
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
  height: "60px",
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
  fontStyle: "normal",
  fontWeight: "450",
  fontSize: 17,
  color: "#000",
  height: "70%",
});
