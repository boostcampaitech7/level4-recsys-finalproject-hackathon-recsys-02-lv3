import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUserId } from "~/utils/userInfoContext";
import { useMutation } from "@tanstack/react-query";
import {
  postOnboardingMutation,
  postOnboardingSelectMutation,
  userEmbeddingQuery,
} from "~/remotes";
import { SelectTags } from "./SelectTags";
import { SelectTracks } from "./SelectTracks";
import { OnboardingSelectItemType, TrackOnboardingSchema } from "~/remotes/dio";
import { client } from "~/libs/react-query";

export type SelectTrackType = [
  TrackOnboardingSchema[],
  TrackOnboardingSchema[]
];

export const Component = () => {
  const navigate = useNavigate();
  const id = useUserId();

  const [step, setStep] = useState<"tag" | "track">("tag");
  const [trackList, setTrackList] = useState<SelectTrackType>();
  const { mutateAsync: mutateSelectTag } = useMutation(postOnboardingMutation);
  const { mutateAsync: mutateSelectTrack } = useMutation(
    postOnboardingSelectMutation
  );

  const handleSubmitTags = async (selectedTags: number[]) => {
    const result = await mutateSelectTag({ user_id: id, tags: selectedTags });
    setTrackList([result.items1, result.items2]);
    setStep("track");
  };

  const handleSubmitTracks = async (
    selectedTracks: OnboardingSelectItemType[]
  ) => {
    await mutateSelectTrack({ user_id: id, items: selectedTracks });
    await client.invalidateQueries(userEmbeddingQuery(id).queryKey);
    navigate("/home");
  };

  return (
    <>
      {step === "tag" && <SelectTags onSubmit={handleSubmitTags} />}
      {step === "track" && trackList && (
        <SelectTracks trackList={trackList} onSubmit={handleSubmitTracks} />
      )}
    </>
  );
};
