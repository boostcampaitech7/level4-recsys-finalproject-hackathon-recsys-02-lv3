import { OnboardingSelectItemType, TrackOnboardingSchema } from "~/remotes/dio";
import { SelectTrackType } from ".";
import { ComponentProps, useState } from "react";
import { css } from "@emotion/react";
import { FixedButton } from "~/components/Button";
import { Spacing } from "~/components/Spacing";
import { MobilePadding } from "~/components/MobilePadding";
import { Title } from "~/components/Title";
import { useLoading } from "~/utils/useLoading";

export const SelectTracks = ({
  trackList,
  onSubmit,
}: {
  trackList: SelectTrackType;
  onSubmit: (value: OnboardingSelectItemType[]) => Promise<void>;
}) => {
  const [isLoading, startLoading] = useLoading();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selected, setSelected] = useState<OnboardingSelectItemType[]>([]);
  const [track1, track2] = trackList;

  return (
    <MobilePadding>
      <Spacing size={24} />
      <Title>
        취향과 더 가까운
        <br />
        음악을 선택해주세요
      </Title>
      <Spacing size={40} />
      <TracksPair
        tracksPair={[track1[currentIndex], track2[currentIndex]]}
        onSelect={(concatValue: OnboardingSelectItemType[]) => {
          setSelected([...selected, ...concatValue]);
          if (currentIndex < 9) setCurrentIndex(currentIndex + 1);
        }}
      />
      <Spacing size={40} />
      {currentIndex === 9 && (
        <FixedButton
          onClick={() => startLoading(onSubmit(selected))}
          disabled={selected.length < 20}
          loading={isLoading}
        >
          가게 무드 완성하기
        </FixedButton>
      )}
    </MobilePadding>
  );
};

const TracksPair = ({
  tracksPair: [track1, track2],
  onSelect,
}: {
  tracksPair: [TrackOnboardingSchema, TrackOnboardingSchema];
  onSelect: (concatValue: OnboardingSelectItemType[]) => void;
}) => {
  const getNewValue = (positive: 0 | 1): OnboardingSelectItemType[] => {
    return [
      {
        track_id: track1.track_id,
        action: positive ? "negative" : "positive",
        process: "onboarding",
      },
      {
        track_id: track2.track_id,
        action: positive ? "positive" : "negative",
        process: "onboarding",
      },
    ];
  };

  return (
    <>
      <TrackContainer track={track1} onClick={() => onSelect(getNewValue(0))} />
      <TrackContainer track={track2} onClick={() => onSelect(getNewValue(1))} />
    </>
  );
};

const TrackContainer = ({
  track,
  ...props
}: { track: TrackOnboardingSchema } & ComponentProps<"div">) => {
  return (
    <div css={trackImageStyle} {...props}>
      <img
        src={track.track_img_url}
        css={css({
          width: "100%",
          position: "absolute",
          objectFit: "cover",
        })}
      />
      <div css={trackInfoStyle}>
        <span>{track.track_name}</span>
        <span>{track.artists.map((v) => v.artist_name).join(", ")}</span>
        <span>{`#${track.tags}`}</span>
      </div>
    </div>
  );
};

const trackImageStyle = css({
  height: 250,
  width: "100%",
  display: "flex",
  flexDirection: "column",
  alignItems: "flex-end",
  justifyContent: "flex-end",
  paddingBottom: 20,
  position: "relative",
});

const trackInfoStyle = css({
  zIndex: 2,
  display: "flex",
  flexDirection: "column",
  backgroundColor: "#000",
  paddingRight: 10,
});
