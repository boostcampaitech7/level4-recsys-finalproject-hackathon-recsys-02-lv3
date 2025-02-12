export interface BaseResponse<T> {
  message: string;
  items: T;
}

export interface PlayListSchema {
  playlist_id: string;
  playlist_name: string;
  playlist_img_url: string;
}

/**
 * playlist tracks
 */
export interface Artist {
  artist_name: string;
}
export interface TrackSchema {
  track_id: string;
  track_name: string;
  track_img_url: string;
  artists: Artist[];
  description: string;
}

/**
 * playlist에 track 추가 request
 */
export type PostTrackRequest = Pick<TrackSchema, "track_name" | "artists">;

export interface TrackOnboardingSchema extends TrackSchema {
  tags: number[];
}

export interface OnboardingSelectItemType {
  track_id: string;
  process: "onboarding";
  action: "positive" | "negative";
}

export interface PostOcrImageResponse {
  track_id: number;
  artists: string[];
  track_img_url: string;
  description: string;
}

export interface OcrTrackRequest {
  track_name: string;
  artist_name: string;
}
