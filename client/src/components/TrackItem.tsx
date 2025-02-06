import { css } from "@emotion/react";
import { CheckCircle } from "~/assets/svg/CheckCircle";
import { PlusCircle } from "~/assets/svg/PlusCircle";
import { Spacing } from "~/components/Spacing";

export const TrackItem = ({
  trackImage,
  trackName,
  artistName,
  description,
  onSelectChange,
  selected,
  rightAddonColor,
}: {
  trackImage?: string;
  trackName: string;
  artistName: string;
  description: string;
  onSelectChange: () => void;
  selected: boolean;
  rightAddonColor?: string;
}) => {
  return (
    <>
      <Spacing size={10} />
      <div
        css={css({
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          width: "100%",
          height: 60,
        })}
      >
        <div css={css({ display: "flex", alignItems: "center" })}>
          {trackImage && (
            <img
              src={trackImage}
              width={60}
              height={60}
              css={css({ marginRight: 12 })}
            />
          )}
          <div
            css={css({
              display: "flex",
              flexDirection: "column",
            })}
          >
            <span>{trackName}</span>
            <span css={css({ fontSize: 12 })}>{artistName}</span>
            <span css={css({ color: "#D82929" })}>{description}</span>
          </div>
        </div>
        <button
          css={css({ flexShrink: 0, marginLeft: 6 })}
          onClick={onSelectChange}
        >
          {selected ? (
            <CheckCircle color={rightAddonColor} />
          ) : (
            <PlusCircle color={rightAddonColor} />
          )}
        </button>
      </div>
      <Spacing size={10} />
    </>
  );
};
