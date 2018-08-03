import Vue from 'vue'
import Router from 'vue-router'
import Camera from '@/components/Camera'
import Dance from '@/components/Dance'
Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Camera',
      component: Camera
    },
    {
      path: '/dance',
      name: 'Dance',
      component: Dance
    }
  ]
})
