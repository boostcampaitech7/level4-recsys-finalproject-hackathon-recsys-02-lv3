import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { FixedButton } from "~/components/Button";
import { Description } from "~/components/Description";
import { MobilePadding } from "~/components/MobilePadding";
import { Spacing } from "~/components/Spacing";
import { Title } from "~/components/Title";
import { TrackItem } from "~/components/TrackItem";
import { postOcrTrackMutation } from "~/remotes";
import { OcrTrackRequest, TrackSchema } from "~/remotes/dio";
import { useLoading } from "~/utils/useLoading";
import { useUserId } from "~/utils/userInfoContext";

export const Ocr트랙리스트 = ({
  ocrTracks,
  onReceiveCandidates,
}: {
  ocrTracks: OcrTrackRequest[];
  onReceiveCandidates: (candidates: TrackSchema[]) => void;
}) => {
  const [selected, setSelected] = useState<OcrTrackRequest[]>(ocrTracks);
  const { mutateAsync } = useMutation(postOcrTrackMutation);
  const id = useUserId();
  const [isLoading, startLoading] = useLoading();

  const handleSelectChange = (track: OcrTrackRequest) => {
    setSelected((prevSelected) => {
      if (prevSelected.some((item) => item.track_name === track.track_name)) {
        return prevSelected.filter(
          (item) => item.track_name !== track.track_name
        );
      } else {
        return [...prevSelected, track];
      }
    });
  };

  const getPayload = (): OcrTrackRequest[] => {
    return ocrTracks
      .filter((track: OcrTrackRequest) =>
        selected.some(
          (selectedTrack) => selectedTrack.track_name === track.track_name
        )
      )
      .map((v: OcrTrackRequest) => ({
        track_name: v.track_name,
        artist_name: v.artist_name,
      }));
  };

  const handleSubmit = async () => {
    const result = await mutateAsync({
      user_id: id,
      items: getPayload(),
    });
    onReceiveCandidates(result);
  };

  return (
    <MobilePadding>
      <Spacing size={40} />
      <Title>
        Upstage OCR API
        <br />
        플레이리스트 인식 결과
      </Title>
      <Spacing size={20} />
      <Description>
        잘못되었거나 필요하지 않은 트랙을 제거하고
        <br />
        '다음으로' 버튼을 눌러주세요
      </Description>
      <Spacing size={20} />
      {ocrTracks?.map((v: OcrTrackRequest) => (
        <TrackItem
          key={v.track_name}
          trackName={v.track_name}
          artistName={v.artist_name}
          onSelectChange={() => handleSelectChange(v)}
          selected={selected.some(
            (selectedItem) => v.track_name === selectedItem.track_name
          )}
          rightAddonColor="#5b52ff"
        />
      ))}
      <Spacing size={20} />
      <FixedButton
        backgroundColor="#5b52ff"
        onClick={() => startLoading(handleSubmit())}
        loading={isLoading}
        disabled={selected.length === 0}
      >
        다음으로
      </FixedButton>
    </MobilePadding>
  );
};
