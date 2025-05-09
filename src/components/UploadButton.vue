<template>
  <div class="flex items-center" :style="{ width: boardWidth + 'px' }">
    <button class="shadcn-btn mr-2 flex-1" style="max-width:40%">
      <label class="w-full h-full flex items-center justify-center cursor-pointer">
        上传棋盘
        <input type="file" accept="image/*" class="hidden" @change="onFileChange" />
      </label>
    </button>
    <div class="flex-4" style="flex:3; max-width:60%">
      <div class="progress-bar-bg">
        <div class="progress-bar-fg" :style="{ width: progress + '%' }"></div>
        <span class="progress-text">{{ progressText }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import axios from 'axios'
const emit = defineEmits(['board-parsed'])

const showProgress = ref(false)
const progress = ref(0)
const progressText = ref('')
let timer = null
const boardWidth = 500 // 10*50，和棋盘宽度一致

// 进度条状态：0-未开始，1-进行中，2-有解，3-无解
const progressStatus = ref(0)

async function onFileChange(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  // 1. 先请求图片解析接口
  const res = await axios.post('/api/parse_board_image', formData)
  // 返回 { board, typeImgs }
  emit('board-parsed', { board: res.data.board, typeImgs: res.data.typeImgs })
  // 进度条重置
  progress.value = 0
  progressStatus.value = 1
  progressText.value = '求解中'
  showProgress.value = true
  // 5秒进度条动画
  let start = Date.now()
  timer && clearInterval(timer)
  timer = setInterval(() => {
    const elapsed = Date.now() - start
    progress.value = Math.min(100, Math.floor(elapsed / 50)) // 5s 到 100
    if (progress.value >= 100) {
      clearInterval(timer)
    }
  }, 50)
}

// 提供给父组件控制进度条和文字
function setProgress(val, status = null) {
  progress.value = val
  if (status !== null) progressStatus.value = status
  if (val >= 100) {
    timer && clearInterval(timer)
    if (progressStatus.value === 2) {
      progressText.value = '求解完毕'
    } else if (progressStatus.value === 3) {
      progressText.value = '此棋盘无解'
    }
  }
}

watch(progressStatus, (val) => {
  if (val === 1) progressText.value = '求解中'
  if (val === 2) progressText.value = '求解完毕'
  if (val === 3) progressText.value = '此棋盘无解'
})

onMounted(() => {
  progressText.value = ''
})

defineExpose({ setProgress, boardWidth })
</script>

<style scoped>
.progress-bar-bg {
  width: 100%;
  height: 20px;
  background: #eee;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}
.progress-bar-fg {
  height: 100%;
  background: linear-gradient(90deg, #4f8cff, #42d392);
  transition: width 0.2s;
  position: absolute;
  left: 0;
  top: 0;
  z-index: 1;
}
.progress-text {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
  z-index: 2;
  color: #333;
  pointer-events: none;
  text-shadow: 0 1px 4px #fff, 0 -1px 4px #fff, 1px 0 4px #fff, -1px 0 4px #fff;
}
.shadcn-btn {
  border: 1.5px solid #222;
  box-shadow: 0 2px 8px 0 rgba(0,0,0,0.10);
  background: #fff;
  color: #222;
  transition: box-shadow 0.2s, border 0.2s, color 0.2s;
}
.shadcn-btn:hover {
  box-shadow: 0 4px 16px 0 rgba(0,0,0,0.18);
  border-color: #000;
  color: #000;
}
</style>
