import { getResult, uploadImageToOCR } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";

/** Upload and poll every two seconds until done or failed. Job selection is not restored after refresh. */
export const useOcrJob = () => {
  const [jobId, setJobId] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const uploadInProgress = useRef(false);

  // Stop polling on request errors or completed and failed jobs.
  const jobQuery = useQuery({
    queryKey: ["ocrJob", jobId],
    queryFn: () => getResult(jobId!),
    enabled: jobId !== null,
    retry: 1,
    refetchInterval: (currentQuery) => {
      if (currentQuery.state.status === "error") {
        return false;
      }

      const status = currentQuery.state.data?.status ?? "";

      if (status === "done" || status === "failed") {
        return false;
      }

      return 2000;
    },
  });

  /** Prevent duplicate uploads and start polling with the returned job ID. */
  const handleUpload = async (file: File) => {
    if (uploadInProgress.current) {
      return;
    }

    uploadInProgress.current = true;
    setIsUploading(true);
    setUploadError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const uploadedJob = await uploadImageToOCR(formData);
      setJobId(uploadedJob.job_id);
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Upload failed. Please try again.",
      );
    } finally {
      setIsUploading(false);
      uploadInProgress.current = false;
    }
  };

  const resetJob = () => {
    setJobId(null);
    setUploadError("");
  };

  return {
    jobId,
    uploading: isUploading,
    handleUpload,
    status: jobQuery.data?.status ?? (jobId === null ? "idle" : "queued"),
    resultData: jobQuery.data,
    error: uploadError || jobQuery.error?.message || jobQuery.data?.message || "",
    reset: resetJob,
    refetch: jobQuery.refetch,
  };
};
