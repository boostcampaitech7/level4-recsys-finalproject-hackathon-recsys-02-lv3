import { useState } from "react";
import { Ocr안내문구 } from "./안내문구";
import { Ocr트랙리스트 } from "./트랙리스트";
import { useNavigate, useSearchParams } from "react-router-dom";
import { OcrTrackRequest, PostTrackRequest, TrackSchema } from "~/remotes/dio";
import { CandidatesPageContent } from "~/components/CandidatesPageContent";
import { useMutation } from "@tanstack/react-query";
import { postOcrPlaylistMutation } from "~/remotes";
import { useUserId } from "~/utils/userInfoContext";

export const Component = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [candidates, setCandidates] = useState<TrackSchema[]>([]);
  const id = useUserId();

  const serializedData = searchParams.get("data");
  const ocrTracks: OcrTrackRequest[] = serializedData
    ? JSON.parse(decodeURIComponent(serializedData))
    : [];

  const { mutateAsync } = useMutation(postOcrPlaylistMutation(id));
  const handleSubmit = async (payload: PostTrackRequest[]) => {
    await mutateAsync({
      items: payload,
    });
    navigate("/home");
  };

  if (location.pathname === "/ocr/candidates") {
    return <CandidatesPageContent data={candidates} onSubmit={handleSubmit} />;
  }

  return (
    <>
      {ocrTracks.length > 0 ? (
        <Ocr트랙리스트
          ocrTracks={ocrTracks}
          onReceiveCandidates={(results) => {
            setCandidates(results);
            navigate("/ocr/candidates");
          }}
        />
      ) : (
        <Ocr안내문구 />
      )}
    </>
  );
};
