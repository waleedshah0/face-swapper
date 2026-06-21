// ---- Tab switching -------------------------------------------------------
const tabImage = document.getElementById("tab-image");
const tabVideo = document.getElementById("tab-video");
const panelImage = document.getElementById("panel-image");
const panelVideo = document.getElementById("panel-video");

function activateTab(which) {
  const isImage = which === "image";
  tabImage.classList.toggle("is-active", isImage);
  tabVideo.classList.toggle("is-active", !isImage);
  tabImage.setAttribute("aria-selected", String(isImage));
  tabVideo.setAttribute("aria-selected", String(!isImage));
  panelImage.hidden = !isImage;
  panelVideo.hidden = isImage;
  panelImage.classList.toggle("is-active", isImage);
  panelVideo.classList.toggle("is-active", !isImage);
}
tabImage.addEventListener("click", () => activateTab("image"));
tabVideo.addEventListener("click", () => activateTab("video"));

// ---- Live previews for dropzones (image or video) -------------------------
document.querySelectorAll(".dropzone input[type=file]").forEach((input) => {
  input.addEventListener("change", () => {
    const label = input.closest(".dropzone").querySelector("label");
    const preview = label.querySelector(".dz-preview");
    const file = input.files[0];
    if (!preview || !file) return;

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.add("has-image");
    if (preview.tagName === "VIDEO") preview.load();
  });
});

function showError(el, message) {
  el.textContent = message;
  el.hidden = false;
}
function clearError(el) {
  el.hidden = true;
  el.textContent = "";
}

// ---- Image swap ------------------------------------------------------------
const formImage = document.getElementById("form-image");
const imageError = document.getElementById("image-error");
const imageResult = document.getElementById("image-result");
const imageResultImg = document.getElementById("image-result-img");
const imageDownload = document.getElementById("image-download");

formImage.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError(imageError);
  imageResult.hidden = true;

  const submitBtn = formImage.querySelector(".develop-btn");
  submitBtn.disabled = true;
  submitBtn.querySelector("span").textContent = "Developing…";

  try {
    const formData = new FormData(formImage);
    const res = await fetch("/api/swap/image", { method: "POST", body: formData });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    imageResultImg.src = url;
    imageDownload.href = url;
    imageDownload.download = "face_swap_result.png";
    imageResult.hidden = false;
  } catch (err) {
    showError(imageError, err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector("span").textContent = "Develop image";
  }
});

// ---- Video swap (async job + polling) --------------------------------------
const formVideo = document.getElementById("form-video");
const videoError = document.getElementById("video-error");
const videoProgressWrap = document.getElementById("video-progress");
const reelFill = document.getElementById("reel-fill");
const progressPct = document.getElementById("progress-pct");
const progressStatus = document.getElementById("progress-status");
const videoResult = document.getElementById("video-result");
const videoPlayer = document.getElementById("video-result-player");
const videoDownload = document.getElementById("video-download");

formVideo.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError(videoError);
  videoResult.hidden = true;

  const submitBtn = formVideo.querySelector(".develop-btn");
  submitBtn.disabled = true;
  submitBtn.querySelector("span").textContent = "Submitting…";

  try {
    const formData = new FormData(formVideo);
    const res = await fetch("/api/swap/video", { method: "POST", body: formData });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }

    const { job_id } = await res.json();
    videoProgressWrap.hidden = false;
    submitBtn.querySelector("span").textContent = "Develop video";
    submitBtn.disabled = false;
    await pollJob(job_id);
  } catch (err) {
    showError(videoError, err.message);
    submitBtn.disabled = false;
    submitBtn.querySelector("span").textContent = "Develop video";
  }
});

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error("Lost track of that job.");
        const job = await res.json();

        reelFill.style.width = `${job.progress}%`;
        progressPct.textContent = Math.round(job.progress);
        progressStatus.textContent = job.status;

        if (job.status === "done") {
          clearInterval(interval);
          videoPlayer.src = job.download_url;
          videoDownload.href = job.download_url;
          videoDownload.download = "face_swap_result.mp4";
          videoResult.hidden = false;
          resolve();
        } else if (job.status === "failed") {
          clearInterval(interval);
          showError(videoError, job.error || "Video swap failed.");
          reject(new Error(job.error || "Video swap failed."));
        }
      } catch (err) {
        clearInterval(interval);
        showError(videoError, err.message);
        reject(err);
      }
    }, 2000);
  });
}
