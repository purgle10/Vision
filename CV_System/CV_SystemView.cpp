
// CV_SystemView.cpp : CCV_SystemView 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "CV_System.h"


#include "AxisState.h"
#include "MainFrm.h"
#include "CameraDS.h"

#include "CV_SystemDoc.h"
#include "CV_SystemView.h"
#include "ImageJudge.h"
#endif

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CCV_SystemView

IMPLEMENT_DYNCREATE(CCV_SystemView, CFormView)

BEGIN_MESSAGE_MAP(CCV_SystemView, CFormView)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()

	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_START, &CCV_SystemView::OnBnClickedStart)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_IMWAIT, &CCV_SystemView::OnBnClickedImwait)
	ON_WM_CLOSE()
	ON_COMMAND(IDM_AxisState, &CCV_SystemView::OnAxisstate)
	ON_BN_CLICKED(IDC_SAVEIMG, &CCV_SystemView::OnClickedSaveimg)
	ON_BN_CLICKED(IDC_CHOOSEREF, &CCV_SystemView::OnClickedChooseref)
	ON_COMMAND(ID_MSE, &CCV_SystemView::OnMse)
	ON_COMMAND(ID_PSNR, &CCV_SystemView::OnPsnr)
	ON_COMMAND(ID_SSIM, &CCV_SystemView::OnSsim)
	ON_COMMAND(ID_CV, &CCV_SystemView::OnCv)
	ON_COMMAND(ID_NR, &CCV_SystemView::OnNr)
	ON_COMMAND(ID_HELP, &CCV_SystemView::OnHelp)
END_MESSAGE_MAP()


// CCV_SystemView 构造/析构

CCV_SystemView::CCV_SystemView()
	: CFormView(CCV_SystemView::IDD)
{
	// TODO: 在此处添加构造代码
	pStc=NULL;
	pDC=NULL;
	img = NULL;
	cap = NULL;
	pAxisDlg=NULL;
}

CCV_SystemView::~CCV_SystemView()
{
	m_CvvImage.Destroy();
	pAxisDlg = NULL;
}

void CCV_SystemView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, m_CBNCamList);
}

BOOL CCV_SystemView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: 在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式

	return CFormView::PreCreateWindow(cs);
}

void CCV_SystemView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	//m_nMapMode = -1;
	ResizeParentToFit();
	pStc=(CStatic *)GetDlgItem(IDC_IMSHOW);//IDC_VIEW为Picture控件ID
	pStc->GetClientRect(rect);//将CWind类客户区的坐标点传给矩形
	pDC=pStc->GetDC(); //得到Picture控件设备上下文
	hDC=pDC->GetSafeHdc(); //得到控件设备上下文的句柄 


	m_nCamCount = CCameraDS::CameraCount();
	if( m_nCamCount < 1 )
	{
		AfxMessageBox(_T("请插入至少1个摄像头！"));
		//return -1;
	}
	// 在组合框CamList中添加摄像头名称的字符串
	char camera_name[1024];
	char istr[25];
	CString camstr;
	for(int i=0; i < m_nCamCount; i++)
	{  
		int retval = CCameraDS::CameraName(i, camera_name, sizeof(camera_name) );

		sprintf_s(istr, " # %d", i);
		strcat_s( camera_name, istr );  
		camstr = camera_name;
		if(retval >0)
			m_CBNCamList.AddString(camstr);
		else
			AfxMessageBox(_T("不能获取摄像头的名称"));
	}
	camstr.ReleaseBuffer();
}

void CCV_SystemView::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CCV_SystemView::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CCV_SystemView 诊断

#ifdef _DEBUG
void CCV_SystemView::AssertValid() const
{
	CFormView::AssertValid();
}

void CCV_SystemView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CCV_SystemDoc* CCV_SystemView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCV_SystemDoc)));
	return (CCV_SystemDoc*)m_pDocument;
}
#endif //_DEBUG


// CCV_SystemView 消息处理程序

//void CCV_SystemView::OnSize(UINT nType, int cx, int cy)
//{
//	CFormView::OnSize(nType, cx, cy);
//	//MessageBox(_T("MAXIMIZED!"));
//	// TODO: 在此处添加消息处理程序代码
//	//if (nType==3)
//	//{
//	//	MessageBox(_T("MAXIMIZED!"));
//	//}
//	//if (nType==SIZE_MAXIMIZED)
//	//{
//	//	MessageBox(_T("MAXIMIZED!"));
//	//}
//	//if (::IsZoomed(m_hWnd))
//	//{
//	//	MessageBox(_T("MAXIMIZED!"));
//	//}
//	//ImageShow();
//}


void CCV_SystemView::OnBnClickedStart()
{
	// TODO: 在此添加控件通知处理程序代码
	//MessageBox(_T("OK!"));
	ImageShow();
}

void CCV_SystemView::ImageShow()
{
	if (!cap.isOpened())
	{
		cap.open(0);
	}
	if (!cap.isOpened())
	{
		AfxMessageBox(_T("无法获得摄像头!"));
		return;
	}
	SetTimer(1, 25, NULL);
}

void CCV_SystemView::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	if (1==nIDEvent)
	{
		cap >> frame;
		img = &IplImage(frame);
		//CFrameWnd * pFwnd = GetParentFrame();
		//pFwnd->GetMenu()->GetSubMenu(0)->EnableMenuItem(1, MF_BYPOSITION | MF_DISABLED | MF_GRAYED);
		theApp.GetMainWnd()->GetMenu()->GetSubMenu(4)->EnableMenuItem(1, MF_BYPOSITION | MF_DISABLED | MF_GRAYED);
		//CvvImage m_CvvImage;
		m_CvvImage.CopyOf(img, 1);
		m_CvvImage.DrawToHDC(hDC, &rect);
	}

	CFormView::OnTimer(nIDEvent);
}


void CCV_SystemView::OnBnClickedImwait()
{
	// TODO: 在此添加控件通知处理程序代码
	KillTimer(1);
}


void CCV_SystemView::OnClose()
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	CFormView::OnClose();
}


void CCV_SystemView::OnAxisstate()
{
	// TODO: 在此添加命令处理程序代码
	pAxisDlg = new CAxisState();
	pAxisDlg->Create(IDD_AxisState, this);
	pAxisDlg->ShowWindow(SW_SHOW);
}


void CCV_SystemView::OnClickedSaveimg()
{
	CString strFilter = _T("*.bmp|*.bmp|*.jpg|*.jpg|*.png|*.png||");
	CFileDialog dlg(FALSE, _T("bmp"), _T("save.bmp"), NULL, strFilter);
	if (dlg.DoModal() != IDOK)
		return;

	CString strFileName;
	CString strExtension;
	strFileName = dlg.m_ofn.lpstrFile;
	strExtension = dlg.m_ofn.lpstrFilter;
	USES_CONVERSION;
	std::string ss(W2A(strFileName)); //under UNICODE, use the WideCharToMutilByte
	cv::Mat saveImg = frame;
	cv::imwrite(ss, saveImg);
}


void CCV_SystemView::OnClickedChooseref()
{
	// TODO:  在此添加控件通知处理程序代码
	CString strFilter = _T("*.bmp|*.bmp|*.jpg|*.jpg|*.png|*.png||");
	CFileDialog dlg(TRUE, _T("bmp"), _T("ref.bmp"), NULL, strFilter);
	if (dlg.DoModal() != IDOK)
		return;

	CString strFileName;
	CString strExtension;
	strFileName = dlg.m_ofn.lpstrFile;
	strExtension = dlg.m_ofn.lpstrFilter;
	USES_CONVERSION;
	std::string ss(W2A(strFileName)); //under UNICODE, use the WideCharToMutilByte
	cv::Mat openImg;
	openImg = cv::imread(ss);
	if (openImg.empty())
	{
		MessageBox(_T("输入参考图像出错！请重新选择！"));
		return;
	}
	ref = openImg;
}


void CCV_SystemView::OnMse()
{
	// TODO:  在此添加命令处理程序代码
	if (ref.empty())
	{
		MessageBox(_T("请先选择参考图像！"));
		return;
	}
	if (frame.empty())
	{
		MessageBox(_T("测试图像不能为空！"));
		return;
	}
	if (frame.size() != ref.size())
	{
		MessageBox(_T("参考图像跟测试大小要一致"));
		return;
	}
	cv::Mat inputRef, inputFrame;
	cv::cvtColor(frame, inputFrame, CV_BGR2GRAY);
	cv::cvtColor(ref, inputRef, CV_BGR2GRAY);
	double Mse = getMSE(inputFrame, inputRef);
	CString strResult;
	strResult.Format(_T("%f"), Mse);
	MessageBox(_T("均方误差为："+strResult));
}


void CCV_SystemView::OnPsnr()
{
	// TODO:  在此添加命令处理程序代码
	if (ref.empty())
	{
		MessageBox(_T("请先选择参考图像！"));
		return;
	}
	if (frame.empty())
	{
		MessageBox(_T("测试图像不能为空！"));
		return;
	}
	if (frame.size() != ref.size())
	{
		MessageBox(_T("参考图像跟测试大小要一致"));
		return;
	}
	cv::Mat inputRef, inputFrame;
	cv::cvtColor(frame, inputFrame, CV_BGR2GRAY);
	cv::cvtColor(ref, inputRef, CV_BGR2GRAY);
	double psnr = getPSNR(inputFrame, inputRef);
	CString strResult;
	strResult.Format(_T("%f"), psnr);
	MessageBox(_T("峰值信噪比为：" + strResult));
}


void CCV_SystemView::OnSsim()
{
	// TODO:  在此添加命令处理程序代码
	if (ref.empty())
	{
		MessageBox(_T("请先选择参考图像！"));
		return;
	}
	if (frame.empty())
	{
		MessageBox(_T("测试图像不能为空！"));
		return;
	}
	if (frame.size() != ref.size())
	{
		MessageBox(_T("参考图像跟测试大小要一致"));
		return;
	}
	cv::Mat inputRef, inputFrame;
	cv::cvtColor(frame, inputFrame, CV_BGR2GRAY);
	cv::cvtColor(ref, inputRef, CV_BGR2GRAY);
	double ssim = getSSIM(inputFrame, inputRef);
	CString strResult;
	strResult.Format(_T("%f"), ssim);
	MessageBox(_T("结构相似度为：" + strResult));
}


void CCV_SystemView::OnCv()
{
	// TODO:  在此添加命令处理程序代码
	if (frame.empty())
	{
		MessageBox(_T("测试图像不能为空！"));
		return;
	}
	cv::Mat inputFrame;
	cv::cvtColor(frame, inputFrame, CV_BGR2GRAY);
	double CV = getCV(inputFrame);
	CString strResult;
	strResult.Format(_T("%f"), CV);
	MessageBox(_T("边缘强度为：" + strResult));
}


void CCV_SystemView::OnNr()
{
	// TODO:  在此添加命令处理程序代码
	if (frame.empty())
	{
		MessageBox(_T("测试图像不能为空！"));
		return;
	}
	cv::Mat inputFrame;
	cv::cvtColor(frame, inputFrame, CV_BGR2GRAY);
	double nr = getNR(inputFrame);
	CString strResult;
	strResult.Format(_T("%f"), nr);
	MessageBox(_T("噪声率为：" + strResult));
}


void CCV_SystemView::OnHelp()
{
	// TODO:  在此添加命令处理程序代码
	ShellExecute(NULL, TEXT("OPEN"),_T("MCT2008Help.CHM"), NULL, NULL, SW_SHOWNORMAL);
}
