import { api } from "~/libs/api";
import { mutationOptions, queryOptions } from "~/libs/react-query";
import {
  BaseResponse,
  OcrTrackRequest,
  OnboardingSelectItemType,
  PlayListSchema,
  PostOcrImageResponse,
  PostTrackRequest,
  TrackOnboardingSchema,
  TrackSchema,
} from "./dio";

const PLAYLIST_URL = (id: number) => `/api/users/${id}/playlists`;
export const playlistQuery = (id: number) =>
  queryOptions({
    queryKey: [PLAYLIST_URL(id)],
    queryFn: async () =>
      await api.get<BaseResponse<PlayListSchema[]>>(PLAYLIST_URL(id)),
  });

const PLAYLIST_TRACKS_URL = (
  playlistId: string,
  id: number,
  playlistName: string
) =>
  `/api/playlists/${playlistId}/tracks?user_id=${id}&playlist_name=${playlistName}`;
export const playlistTracksQuery = (
  playlistId: string,
  id: number,
  playlistName: string
) =>
  queryOptions({
    queryKey: [PLAYLIST_TRACKS_URL(playlistId, id, playlistName)],
    queryFn: async () =>
      await api.get<TrackSchema[]>(
        PLAYLIST_TRACKS_URL(playlistId, id, playlistName)
      ),
  });

const POST_TRACK_URL = (playlistId: string, id: number) =>
  `/api/playlists/${playlistId}/tracks?user_id=${id}`;
export const postTrackMutation = (playlistId: string, id: number) =>
  mutationOptions({
    mutationFn: async (payload: { items: PostTrackRequest[] }) =>
      await api.post(POST_TRACK_URL(playlistId, id), payload),
  });

const POST_ONBOARDING_URL = `/api/onboarding`;
export const postOnboardingMutation = mutationOptions({
  mutationFn: async (payload: { user_id: number; tags: number[] }) =>
    await api.post<{
      items1: TrackOnboardingSchema[];
      items2: TrackOnboardingSchema[];
    }>(POST_ONBOARDING_URL, payload),
});

const POST_ONBOARDING_SELECT_URL = `/api/onboarding/select`;
export const postOnboardingSelectMutation = mutationOptions({
  mutationFn: async (payload: {
    user_id: number;
    items: OnboardingSelectItemType[];
  }) => await api.post(POST_ONBOARDING_SELECT_URL, payload),
});

const POST_OCR_IMAGE_URL = `/api/playlist/image`;
export const postOcrImageMutation = mutationOptions({
  mutationFn: async (payload: { user_id: number; image: File }) => {
    const formData = new FormData();
    formData.append("user_id", String(payload.user_id));
    formData.append("image", payload.image);
    try {
      const response = await api.post<PostOcrImageResponse>(
        POST_OCR_IMAGE_URL,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return JSON.stringify(response);
    } catch (error) {
      console.error("Error posting OCR image:", error);
      throw error;
    }
  },
});

const POST_OCR_TRACK_URL = `/api/playlist/image/tracks`;
export const postOcrTrackMutation = mutationOptions({
  mutationFn: async (payload: { user_id: number; items: OcrTrackRequest[] }) =>
    await api.post<TrackSchema[]>(POST_OCR_TRACK_URL, payload),
});

const POST_OCR_PLAYLIST_URL = (id: number) =>
  `/api/playlist/create?user_id=${id}`;
export const postOcrPlaylistMutation = (id: number) =>
  mutationOptions({
    mutationFn: async (payload: { items: PostTrackRequest[] }) =>
      await api.post(POST_OCR_PLAYLIST_URL(id), payload),
  });

const USER_EMBEDDING_URL = (userId: number) => `/api/user/${userId}/embedding`;
export const userEmbeddingQuery = (userId: number) =>
  queryOptions({
    queryKey: [USER_EMBEDDING_URL(userId)],
    queryFn: async () =>
      await api.get<{ exist: boolean }>(USER_EMBEDDING_URL(userId)),
  });
