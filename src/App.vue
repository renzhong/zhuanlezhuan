<script setup>
import { ref, nextTick } from 'vue'
import UploadButton from './components/UploadButton.vue'
import Board from './components/Board.vue'
import bgImg from './assets/bg.png'
import axios from 'axios'

// 棋盘初始状态：14行10列，全部为0
const emptyBoard = Array.from({ length: 14 }, () => Array(10).fill(0))
const board = ref(emptyBoard)
const actions = ref([])
const currentStep = ref(0)
const typeImgs = ref({})
const boardList = ref([])

// 进度条控制
const uploadBtnRef = ref(null)
const boardWidth = 500 // 10*50

async function handleBoardParsed(data) {
  // 1. 渲染初始棋盘
  board.value = data.board
  typeImgs.value = data.typeImgs
  boardList.value = []
  actions.value = {}
  currentStep.value = 0
  
  // 如果解析失败，直接显示错误状态
  // if (!data.parse_success) {
  //   console.log('解析棋盘失败', data.parse_success)
  //   uploadBtnRef.value?.setProgress(100, 3, '解析棋盘失败')
  //   return
  // }
  
  // 2. 等待棋盘渲染完成后再异步请求求解接口
  await nextTick()
  await Promise.resolve() // 进一步确保渲染
  startSolve(data.board)
}

async function startSolve(boardData) {
  // 启动进度条
  uploadBtnRef.value?.setProgress(0, 1)
  let progressTimer = setInterval(() => {
    if (uploadBtnRef.value) {
      let cur = uploadBtnRef.value.progress
      if (cur < 99) uploadBtnRef.value.setProgress(cur + 1, 1)
    }
  }, 50)
  try {
    const res = await axios.post('/api/solve_board', { board: boardData })
    boardList.value = res.data.board_list || []
    actions.value = boardList.value[0]?.action || {}
    currentStep.value = 0
    if (boardList.value.length > 0) {
      board.value = boardList.value[0].board
    }
    // 进度条直接100%，并根据有无解设置状态
    if (boardList.value.length > 0 && Object.keys(boardList.value[0].action || {}).length > 0) {
      uploadBtnRef.value?.setProgress(100, 2) // 有解
    } else {
      uploadBtnRef.value?.setProgress(100, 3) // 无解
    }
  } finally {
    clearInterval(progressTimer)
  }
}

function goPrev() {
  if (currentStep.value > 0) {
    currentStep.value--
    board.value = boardList.value[currentStep.value].board
    actions.value = boardList.value[currentStep.value].action
  }
}

function goNext() {
  if (currentStep.value < boardList.value.length - 1) {
    currentStep.value++
    board.value = boardList.value[currentStep.value].board
    actions.value = boardList.value[currentStep.value].action
  }
}
</script>

<template>
  <div class="flex flex-col items-center">
    <div class="mb-4" :style="{ width: boardWidth + 'px' }">
      <UploadButton ref="uploadBtnRef" @board-parsed="handleBoardParsed" />
    </div>
    <div class="flex items-center">
      <button class="shadcn-btn mr-2" @click="goPrev" :disabled="currentStep === 0">◀</button>
      <Board
        :board="board"
        :actions="actions"
        :currentStep="currentStep"
        :typeImgs="typeImgs"
        :bgImg="bgImg"
      />
      <button class="shadcn-btn ml-2" @click="goNext" :disabled="currentStep >= boardList.length - 1">▶</button>
    </div>
  </div>
</template>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
.shadcn-btn {
  padding: 0.5rem 1.2rem;
  border-radius: 0.375rem;
  border: 1.5px solid #6366f1; /* 主色调边框 */
  background: #6366f1;          /* 主色调背景 */
  color: #fff;
  font-weight: 500;
  box-shadow: 0 2px 8px 0 rgba(99, 102, 241, 0.15);
  transition: background 0.2s, box-shadow 0.2s, border 0.2s;
  cursor: pointer;
}
.shadcn-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  background: #a5b4fc;
  border-color: #a5b4fc;
  box-shadow: none;
}
.shadcn-btn:not(:disabled):hover {
  background: #4f46e5;
  border-color: #4f46e5;
  box-shadow: 0 4px 16px 0 rgba(99, 102, 241, 0.25);
}
</style>
