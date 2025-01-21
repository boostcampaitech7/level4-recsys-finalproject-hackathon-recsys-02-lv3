import { css } from "@emotion/react";
import Lottie from "lottie-react";
import loading from "~/assets/loading-lottie.json";

export const FullScreenLoader = () => {
  return (
    <div
      css={css({
        position: "fixed",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        top: 0,
        right: 0,
        left: 0,
        bottom: 0,
        width: "100%",
        height: "100%",
        backgroundColor: "#121212",
        zIndex: 99,
      })}
    >
      <Lottie
        animationData={loading}
        loop={true}
        css={css({ height: 120, width: 120 })}
      />
    </div>
  );
};
