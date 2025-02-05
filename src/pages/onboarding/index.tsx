import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUserId } from "~/utils/userInfoContext";
import { AuthGuard } from "~/components/AuthGuard";
import { useMutation } from "@tanstack/react-query";
import {
  postOnboardingMutation,
  postOnboardingSelectMutation,
} from "~/remotes";
import { SelectTags } from "./SelectTags";
import { SelectTracks } from "./SelectTracks";
import { OnboardingSelectItemType, TrackOnboardingSchema } from "~/remotes/dio";

export type SelectTrackType = [
  TrackOnboardingSchema[],
  TrackOnboardingSchema[],
];

const OnBoardingPage = () => {
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

export const Component = () => (
  <AuthGuard>
    <OnBoardingPage />
  </AuthGuard>
);
