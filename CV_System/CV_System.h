
// CV_System.h : CV_System 应用程序的主头文件
//
#pragma once

#ifndef __AFXWIN_H__
	#error "在包含此文件之前包含“stdafx.h”以生成 PCH 文件"
#endif

#include "resource.h"       // 主符号

#include "CvvImage.h"
// CCV_SystemApp:
// 有关此类的实现，请参阅 CV_System.cpp
//

class CCV_SystemApp : public CWinAppEx
{
public:
	CCV_SystemApp();


// 重写
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// 实现
	UINT  m_nAppLook;
	BOOL  m_bHiColorIcons;
	UINT  m_nX;
	UINT  m_nY;
	UINT  m_nHeight;
	UINT  m_nWideth;
	virtual void PreLoadState();
	virtual void LoadCustomState();
	virtual void SaveCustomState();

	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()

};

extern CCV_SystemApp theApp;
