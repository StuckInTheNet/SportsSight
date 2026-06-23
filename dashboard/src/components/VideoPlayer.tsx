import { useCallback, useEffect, useRef, useState } from "react";
import { FastForward, Pause, Play, SkipBack, SkipForward, Video } from "lucide-react";

interface VideoFile {
  name: string;
  size_mb: number;
}

interface Props {
  onTimeUpdate?: (timeMs: number) => void;
  seekToMs?: number;
  /** When fatigue data starts — enables "Skip to action" button */
  dataStartMs?: number;
}

export default function VideoPlayer({ onTimeUpdate, seekToMs, dataStartMs }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    fetch("/api/video")
      .then((r) => r.json())
      .then((data) => {
        setVideos(data);
        if (data.length > 0) {
          // Prefer the annotated video, then 720p, then first
          const pick =
            data.find((v: VideoFile) => v.name.includes("annotated")) ||
            data.find((v: VideoFile) => v.name.includes("720p")) ||
            data[0];
          setSelectedVideo(pick.name);
        }
      })
      .catch(() => {});
  }, []);

  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime * 1000;
    setCurrentTime(t);
    onTimeUpdate?.(t);
  }, [onTimeUpdate]);

  useEffect(() => {
    if (seekToMs !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekToMs / 1000;
    }
  }, [seekToMs]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setPlaying(true);
    } else {
      videoRef.current.pause();
      setPlaying(false);
    }
  };

  const skip = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime += seconds;
  };

  const jumpToData = () => {
    if (!videoRef.current || !dataStartMs) return;
    // Jump to 5 seconds before data starts so user sees the transition
    videoRef.current.currentTime = Math.max(0, dataStartMs / 1000 - 5);
    videoRef.current.play();
    setPlaying(true);
  };

  const fmt = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
  };

  const showSkipButton =
    dataStartMs && dataStartMs > 30000 && currentTime < dataStartMs - 5000;

  if (!selectedVideo) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 flex flex-col items-center justify-center h-64">
        <Video className="w-8 h-8 text-gray-600 mb-2" />
        <p className="text-sm text-gray-500">No video files found</p>
        <p className="text-xs text-gray-600 mt-1">Add .mp4 files to data/raw/videos/</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      {videos.length > 1 && (
        <div className="px-3 pt-3">
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-1 text-xs focus:outline-none"
            value={selectedVideo}
            onChange={(e) => setSelectedVideo(e.target.value)}
          >
            {videos.map((v) => (
              <option key={v.name} value={v.name}>
                {v.name} ({v.size_mb} MB)
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="relative bg-black">
        <video
          ref={videoRef}
          src={`/api/video/${selectedVideo}`}
          className="w-full"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={() => {
            if (videoRef.current) setDuration(videoRef.current.duration * 1000);
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={togglePlay}
        />

        {/* Skip to action overlay */}
        {showSkipButton && (
          <button
            className="absolute bottom-4 right-4 flex items-center gap-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium px-3 py-1.5 rounded-lg shadow-lg transition-colors"
            onClick={jumpToData}
          >
            <FastForward className="w-3.5 h-3.5" />
            Skip to {fmt(dataStartMs!)}
          </button>
        )}
      </div>

      <div className="px-3 py-2">
        <div
          className="h-1.5 bg-gray-700 rounded-full cursor-pointer mb-2 group relative"
          onClick={(e) => {
            if (!videoRef.current || !duration) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            videoRef.current.currentTime = (pct * duration) / 1000;
          }}
        >
          {/* Data start marker */}
          {dataStartMs && duration > 0 && (
            <div
              className="absolute top-0 h-full w-0.5 bg-blue-500 z-10"
              style={{ left: `${(dataStartMs / duration) * 100}%` }}
              title={`Fatigue data starts at ${fmt(dataStartMs)}`}
            />
          )}
          <div
            className="h-full bg-blue-500 rounded-full transition-all group-hover:bg-blue-400"
            style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button className="p-1 hover:bg-gray-800 rounded" onClick={() => skip(-10)}>
              <SkipBack className="w-4 h-4" />
            </button>
            <button className="p-1.5 hover:bg-gray-800 rounded-full" onClick={togglePlay}>
              {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
            <button className="p-1 hover:bg-gray-800 rounded" onClick={() => skip(10)}>
              <SkipForward className="w-4 h-4" />
            </button>
          </div>
          <span className="text-xs text-gray-500">{fmt(currentTime)} / {fmt(duration)}</span>
        </div>
      </div>
    </div>
  );
}
