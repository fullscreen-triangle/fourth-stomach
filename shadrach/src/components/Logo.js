import { motion } from 'framer-motion'
import Link from 'next/link'
import React from 'react'

let MotionLink = motion(Link);

const Logo = () => {
  return (
    <div className='flex flex-col items-center justify-center mt-2'>
      <MotionLink href="/"
        className='flex items-center justify-center rounded-full w-16 h-16 bg-primary/20 border-2 border-primary text-primary
        text-lg font-bold tracking-tight'
        whileHover={{
          backgroundColor: ["rgba(44,168,154,0.2)", "rgba(44,168,154,0.5)", "rgba(88,230,217,0.3)", "rgba(44,168,154,0.2)"],
          borderColor: ["#2ca89a", "#58E6D9", "#2ca89a"],
          transition: { duration: 2, repeat: Infinity }
        }}
      >FS</MotionLink>
    </div>
  )
}

export default Logo
