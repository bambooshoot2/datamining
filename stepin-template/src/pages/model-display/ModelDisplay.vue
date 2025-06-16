<template>
  <div class="diabetes-bg">
    <el-row justify="center" align="middle" class="page-row" style="min-height:100vh;">
      <el-col :xs="24" :sm="20" :md="16" :lg="10" :xl="8">
        <el-card shadow="always" class="diabetes-card">
          <h2 class="title">糖尿病风险智能检测</h2>
          <div class="sub">
            请如实填写检测项，系统将智能预测患糖尿病概率
          </div>
          <el-divider />

          <el-form :model="form" label-width="100px" label-position="top" class="diabetes-form" @submit.prevent>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="性别">
                  <el-select v-model="form.gender" placeholder="请选择" style="width:100%">
                    <el-option label="男性" value="男性"></el-option>
                    <el-option label="女性" value="女性"></el-option>
                    <el-option label="未知" value="未知"></el-option>
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="年龄（0~80岁）">
                  <el-input-number 
                    v-model="form.age" 
                    :min="0" 
                    :max="80" 
                    controls-position="right" 
                    style="width: 100%;" 
                    :step="1" 
                    placeholder="年龄(岁)" />
                </el-form-item>
              </el-col>
            </el-row>

            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="高血压">
                  <el-select v-model="form.hypertension" style="width:100%">
                    <el-option label="有" value="有"></el-option>
                    <el-option label="无" value="无"></el-option>
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="心脏病">
                  <el-select v-model="form.heart_disease" style="width:100%">
                    <el-option label="有" value="有"></el-option>
                    <el-option label="无" value="无"></el-option>
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="吸烟史">
              <el-select v-model="form.smoking_history" placeholder="请选择" style="width:100%">
                <el-option label="从未吸烟" value="从未吸烟"></el-option>
                <el-option label="以前吸烟" value="以前吸烟"></el-option>
                <el-option label="经常吸烟" value="经常吸烟"></el-option>
                <el-option label="最近没有吸烟" value="最近没有吸烟"></el-option>
                <el-option label="最近正在吸烟" value="最近正在吸烟"></el-option>
                <el-option label="无信息" value="无信息"></el-option>
              </el-select>
            </el-form-item>
            
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="BMI（10.01~91.82）">
                  <el-input-number 
                    v-model="form.bmi" 
                    :min="10.01" 
                    :max="91.82" 
                    :step="0.01" 
                    controls-position="right" 
                    style="width: 100%;" 
                    placeholder="体质指数"/>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="糖化血红蛋白(HbA1c 3.5~9.0)">
                  <el-input-number 
                    v-model="form.HbA1c_level" 
                    :min="3.5" 
                    :max="9.0" 
                    :step="0.1" 
                    controls-position="right" 
                    style="width: 100%;" 
                    placeholder="HbA1c"/>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="血糖水平（80~300）">
              <el-input-number
                v-model="form.blood_glucose_level" 
                :min="80"
                :max="300"
                :step="1"
                controls-position="right"
                style="width: 100%;"
                placeholder="血糖水平"/>
            </el-form-item>

            <el-form-item class="form-btns">
              <el-button type="primary" @click="onSubmit" :loading="loading" size="large">提交检测</el-button>
              <el-button @click="onReset" size="large" style="margin-left:15px;">重置</el-button>
            </el-form-item>
          </el-form>

          <el-divider />

          <transition name="fade">
            <el-alert
              v-if="result !== null"
              :type="result === 1 ? 'warning' : 'success'"
              :title="result === 1 ? '有糖尿病风险' : '暂无糖尿病风险'"
              :description="result === 1 ?
                '预测显示，您的糖尿病概率较高，请咨询医生，注意生活方式和饮食管理。' :
                '恭喜您暂无糖尿病风险，请继续保持良好生活习惯！'"
              show-icon
              effect="light"
              class="result-alert"
            />
          </transition>

        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

const form = ref({
  gender: '',
  age: null,
  hypertension: '',
  heart_disease: '',
  smoking_history: '',
  bmi: null,
  HbA1c_level: null,
  blood_glucose_level: null
})
const result = ref(null)
const loading = ref(false)

const onReset = () => {
  form.value = {
    gender: '',
    age: null,
    hypertension: '',
    heart_disease: '',
    smoking_history: '',
    bmi: null,
    HbA1c_level: null,
    blood_glucose_level: null
  }
  result.value = null
}

const onSubmit = async () => {
  for (let key in form.value) {
    if(form.value[key] === null || form.value[key] === '' || typeof form.value[key] === 'undefined') {
      ElMessage.warning('请填写所有检测项')
      return
    }
  }
  const postData = { ...form.value }
  loading.value = true
  try {
    const res = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(postData)
    })
    const data = await res.json()
    result.value = data.diabetes
  } catch (e) {
    ElMessage.error('请求失败: ' + e)
  }
  loading.value = false
}
</script>

<style scoped>
/* ...你之前的样式保持不变... */
.diabetes-bg {
  min-height: 100vh;
  background: linear-gradient(135deg, #eff3fa 0%, #e9f7ff 100%);
}
.page-row {
  min-height: 100vh;
}
.diabetes-card {
  border-radius: 22px;
  box-shadow: 0 10px 24px 0 #d0d6e040;
  padding: 45px 30px 30px 30px;
  margin: 40px 0;
  background: #fff;
}
.title {
  font-weight: 700;
  font-size: 2.2rem;
  color: #31568d;
  margin-bottom: 10px;
}
.sub {
  color: #6e7a90;
  font-size: 1.1rem;
  margin-bottom: 5px;
  line-height: 22px;
}
.diabetes-form {
  margin-top: 18px;
  margin-bottom: 10px;
  /* 提高空间感 */
}
.form-btns {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}
.result-alert {
  margin: 32px 0 0 0;
  font-size: 1.08rem;
}
.fade-enter-active, .fade-leave-active {
  transition: opacity .5s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
</style>
