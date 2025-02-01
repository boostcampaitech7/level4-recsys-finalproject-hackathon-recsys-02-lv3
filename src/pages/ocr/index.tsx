import { css } from "@emotion/react";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AuthGuard } from "~/components/AuthGuard";
import { Button } from "~/components/Button";
import { OcrItem } from "~/components/OcrItem";
import { Spacing } from "~/components/Spacing";
import { postOcrTrackMutation } from "~/remotes";
import { OcrTrackRequest } from "~/remotes/dio";
import { useUserId } from "~/utils/userInfoContext";

const OcrPage = () => {
  const [searchParams] = useSearchParams();
  const serializedData = searchParams.get("data");
  const navigate = useNavigate();
  const [selected, setSelected] = useState<OcrTrackRequest[]>([]);
  const { mutateAsync } = useMutation(postOcrTrackMutation);

  const id = useUserId();
  const ocrTracks = serializedData
    ? JSON.parse(decodeURIComponent(serializedData))
    : [];

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
      .filter(
        (track: OcrTrackRequest) =>
          !selected.some(
            (selectedTrack) => selectedTrack.track_name === track.track_name
          )
      )
      .map((v: OcrTrackRequest) => ({
        track_name: v.track_name,
        artist_name: v.artist_name,
      }));
  };

  return (
    <div>
      <Spacing size={15}></Spacing>
      <div css={css({ textAlign: "center", fontWeight: 600 })}>
        Upstage OCR API
        <br />
        플레이리스트 인식 결과
      </div>
      <Spacing size={10}></Spacing>
      {ocrTracks?.map((v: OcrTrackRequest) => (
        <OcrItem
          key={v.track_name}
          trackName={v.track_name}
          artistName={v.artist_name}
          onSelectChange={() => handleSelectChange(v)}
          selected={selected.some(
            (selectedItem) => v.track_name === selectedItem.track_name
          )}
        />
      ))}
      <Button
        backgroundColor="#B62121"
        onClick={async () => {
          await mutateAsync({
            user_id: id,
            items: getPayload(),
          });
          navigate("/playlist/image");
        }}
      >
        플레이리스트 확정하기
      </Button>
    </div>
  );
};

export const Component = () => (
  <AuthGuard>
    <OcrPage />
  </AuthGuard>
);
