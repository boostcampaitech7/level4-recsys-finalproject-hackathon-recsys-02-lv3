import { css } from "@emotion/react";
import { ComponentProps } from "react";
import RefreshIcon from "~/assets/svg/ArrowClockwise.svg";

export const RefreshButton = (props: ComponentProps<"button">) => {
  return (
    <>
      <button {...props} css={refreshButtonStyle} />
      <img src={RefreshIcon} alt="Refresh Icon" />
    </>
  );
};

// https://emotion.sh/docs/css-prop#use-the-css-prop
const refreshButtonStyle = css({
  display: "flex",
  width: "100% - 50px",
  padding: "1px 15px",
  borderRadius: 10,
  height: 30,
  textAlign: "center",
  fontSize: 17,
  fontWeight: 400,
  backgroundColor: "#D9D9D9",
  color: "#121212",
});
