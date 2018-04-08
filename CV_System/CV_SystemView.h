
// CV_SystemView.h : CCV_SystemView 类的接口
//

#pragma once

#include "resource.h"

#include "AxisState.h"
#include "afxwin.h"
#include<opencv2\opencv.hpp>
class CCV_SystemView : public CFormView
{
protected: // 仅从序列化创建
	CCV_SystemView();
	DECLARE_DYNCREATE(CCV_SystemView)

public:
	enum{ IDD = IDD_CV_SYSTEM_FORM };

// 特性
public:
	CCV_SystemDoc* GetDocument() const;

// 操作
public:

// 重写
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	virtual void OnInitialUpdate(); // 构造后第一次调用

// 实现
public:
	virtual ~CCV_SystemView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 生成的消息映射函数
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
//	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnBnClickedStart();
private:
	CRect rect;
	CStatic* pStc; //标识图像显示的Picture控件
	CDC* pDC; //视频显示控件设备上下文
	HDC hDC; //视频显示控件设备句柄
public:

	cv::VideoCapture cap; //视频获取接口
	CvvImage m_CvvImage;
	IplImage* img;
	cv::Mat frame;
	cv::Mat ref;
	CAxisState* pAxisDlg;
public:
	void ImageShow();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnBnClickedImwait();
	afx_msg void OnClose();
	afx_msg void OnAxisstate();
	CComboBox m_CBNCamList;
	int m_nCamCount;
	afx_msg void OnClickedSaveimg();
	afx_msg void OnClickedChooseref();
	afx_msg void OnMse();
	afx_msg void OnPsnr();
	afx_msg void OnSsim();
	afx_msg void OnCv();
	afx_msg void OnNr();
	afx_msg void OnHelp();
};

#ifndef _DEBUG  // CV_SystemView.cpp 中的调试版本
inline CCV_SystemDoc* CCV_SystemView::GetDocument() const
   { return reinterpret_cast<CCV_SystemDoc*>(m_pDocument); }
#endif

