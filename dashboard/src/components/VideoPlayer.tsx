import { useCallback, useEffect, useRef, useState } from "react";
import { Play, Pause, SkipBack, SkipForward, Video } from "lucide-react";

interface VideoFile {
  name: string;
  size_mb: number;
}

interface Props {
  /** Called when video time changes — ms from start */
  onTimeUpdate?: (timeMs: number) => void;
  /** External seek command — set this to jump the video */
  seekToMs?: number;
}

export default function VideoPlayer({ onTimeUpdate, seekToMs }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Fetch available videos
  useEffect(() => {
    fetch("/api/video")
      .then((r) => r.json())
      .then((data) => {
        setVideos(data);
        if (data.length > 0) {
          // Auto-select the 720p video if available, otherwise first
          const pick = data.find((v: VideoFile) => v.name.includes("720p")) || data[0];
          setSelectedVideo(pick.name);
        }
      })
      .catch(() => {});
  }, []);

  // Handle time updates from video
  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime * 1000;
    setCurrentTime(t);
    onTimeUpdate?.(t);
  }, [onTimeUpdate]);

  // External seek
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

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  if (!selectedVideo) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 flex flex-col items-center justify-center h-64">
        <Video className="w-8 h-8 text-gray-600 mb-2" />
        <p className="text-sm text-gray-500">No video files found</p>
        <p className="text-xs text-gray-600 mt-1">
          Add .mp4 files to data/raw/videos/
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      {/* Video selector */}
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

      {/* Video element */}
      <div className="relative bg-black">
        <video
          ref={videoRef}
          src={`/api/video/${selectedVideo}`}
          className="w-full"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={() => {
            if (videoRef.current) {
              setDuration(videoRef.current.duration * 1000);
            }
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={togglePlay}
        />
      </div>

      {/* Controls */}
      <div className="px-3 py-2">
        {/* Progress bar */}
        <div
          className="h-1.5 bg-gray-700 rounded-full cursor-pointer mb-2 group"
          onClick={(e) => {
            if (!videoRef.current || !duration) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            videoRef.current.currentTime = (pct * duration) / 1000;
          }}
        >
          <div
            className="h-full bg-blue-500 rounded-full transition-all group-hover:bg-blue-400"
            style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              className="p-1 hover:bg-gray-800 rounded"
              onClick={() => skip(-10)}
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              className="p-1.5 hover:bg-gray-800 rounded-full"
              onClick={togglePlay}
            >
              {playing ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5" />
              )}
            </button>
            <button
              className="p-1 hover:bg-gray-800 rounded"
              onClick={() => skip(10)}
            >
              <SkipForward className="w-4 h-4" />
            </button>
          </div>

          <span className="text-xs text-gray-500">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      </div>
    </div>
  );
}
