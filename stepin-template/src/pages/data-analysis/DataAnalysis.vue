<template>
  <div class="p-4 max-w-xl mx-auto">
    <h2 class="text-xl font-bold mb-4 text-center">糖尿病风险预测</h2>
    <form @submit.prevent="predictDiabetes" class="space-y-4">
      
      <!-- 性别 -->
      <div class="flex items-center">
        <label class="w-32">性别：</label>
        <select v-model="form.gender" required class="flex-1 border px-2 py-1 rounded">
          <option value="1">男</option>
          <option value="0">女</option>
          <option value="2">其他</option>
        </select>
      </div>

      <!-- 年龄 -->
      <div class="flex items-center">
        <label class="w-32">年龄：</label>
        <input v-model.number="form.age" type="number" required class="flex-1 border px-2 py-1 rounded" />
      </div>

      <!-- 高血压 -->
      <div class="flex items-center">
        <label class="w-32">是否患有高血压：</label>
        <select v-model.number="form.hypertension" required class="flex-1 border px-2 py-1 rounded">
          <option :value="0">否</option>
          <option :value="1">是</option>
        </select>
      </div>

      <!-- 心脏病 -->
      <div class="flex items-center">
        <label class="w-32">是否患有心脏病：</label>
        <select v-model.number="form.heart_disease" required class="flex-1 border px-2 py-1 rounded">
          <option :value="0">否</option>
          <option :value="1">是</option>
        </select>
      </div>

      <!-- 吸烟史 -->
      <div class="flex items-center">
        <label class="w-32">吸烟史：</label>
        <select v-model="form.smoking_history" required class="flex-1 border px-2 py-1 rounded">
          <option :value="4">never</option>
          <option :value="0">no info</option>
          <option :value="3">former</option>
          <option :value="5">not current</option>
          <option :value="2">ever</option>
          <option :value="1">current</option>
        </select>
      </div>

      <!-- BMI -->
      <div class="flex items-center">
        <label class="w-32">BMI：</label>
        <input v-model.number="form.bmi" type="number" step="0.1" required class="flex-1 border px-2 py-1 rounded" />
      </div>

      <!-- HbA1c -->
      <div class="flex items-center">
        <label class="w-32">HbA1c 值：</label>
        <input v-model.number="form.HbA1c_level" type="number" step="0.1" required class="flex-1 border px-2 py-1 rounded" />
      </div>

      <!-- 血糖 -->
      <div class="flex items-center">
        <label class="w-32">血糖值：</label>
        <input v-model.number="form.blood_glucose_level" type="number" required class="flex-1 border px-2 py-1 rounded" />
      </div>

      <!-- 提交按钮 -->
      <div>
        <button type="submit" class="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded">
          提交预测
        </button>
      </div>

      <!-- 结果显示 -->
      <div v-if="result !== null" class="mt-4 text-lg font-semibold text-center">
        预测结果：
        <span :class="result ? 'text-red-500' : 'text-green-600'">
          {{ result ? '可能患有糖尿病' : '未患糖尿病' }}
        </span>
      </div>
    </form>
  </div>
</template>


<script setup>
import { ref } from 'vue'
import axios from 'axios'

const form = ref({
  gender: 1,
  age: 30,
  hypertension: 0,
  heart_disease: 0,
  smoking_history: 0,
  bmi: 25.0,
  HbA1c_level: 5.5,
  blood_glucose_level: 120,
})

const result = ref(1)

const predictDiabetes = async () => {
    
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/predict', form.value)
    result.value = response.data.diabetes === 1
  } catch (error) {
    console.error('预测失败:', error)
    alert('预测请求失败')
  }
}
</script>
