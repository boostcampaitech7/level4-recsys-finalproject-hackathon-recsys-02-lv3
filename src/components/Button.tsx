import { css } from "@emotion/react";
import { ComponentProps } from "react";

export const Button = (props: ComponentProps<"button">) => {
  return <button {...props} css={buttonStyle} />;
};

// https://emotion.sh/docs/css-prop#use-the-css-prop
const buttonStyle = css({
  width: "100%",
  borderRadius: 8,
  height: 56,
  backgroundColor: "#1ED760",
});
